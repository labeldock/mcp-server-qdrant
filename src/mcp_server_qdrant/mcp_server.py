import json
import logging
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.permissions import (
    DELETE,
    PERM_WORD,
    READ,
    WRITE,
    union_permissions,
)
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings

        # Parse the per-collection access control directives from COLLECTION_NAME.
        # An empty mapping means "no whitelist" (legacy behaviour: any collection,
        # gated only by QDRANT_READ_ONLY).
        self._collections = qdrant_settings.collection_access()
        self._union_perms = union_permissions(self._collections)
        self._default_collection = qdrant_settings.default_collection_name()
        self._exposed_tools: list[str] = []

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            self._default_collection,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
        )

        # Build the shared-secret gate. When MCP_PASSWORD is set, the whole MCP
        # endpoint (including tools/list and their descriptions) requires a matching
        # bearer token; the /health custom route stays public.
        auth = None
        if qdrant_settings.mcp_password:
            auth = StaticTokenVerifier(
                tokens={
                    qdrant_settings.mcp_password: {
                        "client_id": "mcp-server-qdrant",
                        "scopes": ["mcp"],
                    }
                },
                required_scopes=["mcp"],
            )

        super().__init__(name=name, instructions=instructions, auth=auth, **settings)

        self.setup_tools()
        self._print_startup_summary()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_id = f"<id>{entry.id}</id>" if entry.id else ""
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry>{entry_id}<content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    # ------------------------------------------------------------------ #
    # Access control helpers
    # ------------------------------------------------------------------ #
    def _expose(self, perm: str) -> bool:
        """Whether a tool requiring ``perm`` should be registered at all."""
        if not self._collections:
            # Legacy mode: read is always available; write/delete follow the global
            # QDRANT_READ_ONLY flag.
            if perm == READ:
                return True
            return not self.qdrant_settings.read_only
        return perm in self._union_perms

    def _check_access(self, collection_name: str | None, perm: str) -> str | None:
        """
        Validate a single tool call. Returns an error message to return to the
        caller when access is denied, or ``None`` when the call may proceed.
        """
        if not self._collections:
            return None
        if not collection_name:
            return "Access denied: a collection name is required."
        access = self._collections.get(collection_name)
        if access is None:
            available = ", ".join(self._collections) or "(none)"
            return (
                f"Access denied: collection '{collection_name}' is not served by "
                f"this server. Available collections: {available}."
            )
        if not access.can(perm):
            return (
                f"Access denied: '{PERM_WORD[perm]}' is not permitted on collection "
                f"'{collection_name}' (permissions: {access.perm_string()})."
            )
        return None

    def _access_summary(self, required_perm: str) -> str:
        """A description suffix that documents collections and their permissions."""
        if not self._collections:
            return ""
        listing = "; ".join(
            f"{acc.name} [{', '.join(acc.perm_words()) or 'no access'}]"
            for acc in self._collections.values()
        )
        allowed = [
            name for name, acc in self._collections.items() if acc.can(required_perm)
        ]
        return (
            f"\n\nAccessible collections: {listing}."
            f"\nThis tool requires '{PERM_WORD[required_perm]}' permission; "
            f"allowed collections: {', '.join(allowed) or '(none)'}."
        )

    def _register_tool(self, func, *, name: str, description: str, perm: str) -> None:
        self.tool(func, name=name, description=description + self._access_summary(perm))
        self._exposed_tools.append(name)

    def _print_startup_summary(self) -> None:
        prefix = "[mcp-server-qdrant]"
        if self.qdrant_settings.mcp_password:
            print(
                f"{prefix} Auth: PASSWORD REQUIRED "
                f"(Authorization: Bearer <MCP_PASSWORD>)"
            )
        else:
            print(f"{prefix} Auth: OPEN (no MCP_PASSWORD set)")

        if self._collections:
            print(f"{prefix} Serving {len(self._collections)} collection(s):")
            width = max(len(name) for name in self._collections)
            for acc in self._collections.values():
                words = ", ".join(acc.perm_words()) or "no access"
                print(
                    f"{prefix}   {acc.name.ljust(width)}  "
                    f"{acc.perm_string()}  ({words})"
                )
        else:
            mode = (
                "read-only" if self.qdrant_settings.read_only else "read/write/delete"
            )
            print(
                f"{prefix} Serving: any collection "
                f"(no COLLECTION_NAME whitelist; {mode})"
            )

        exposed = ", ".join(self._exposed_tools) or "(none)"
        print(f"{prefix} Exposed tools: {exposed}")

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            denied = self._check_access(collection_name, WRITE)
            if denied:
                return denied

            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param query_filter: The filter to apply to the query.
            :return: A list of entries found or None.
            """
            denied = self._check_access(collection_name, READ)
            if denied:
                return [denied]

            # Log query_filter
            await ctx.debug(f"Query filter: {query_filter}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Finding results for query {query}")

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        async def update(
            ctx: Context,
            point_id: Annotated[str, Field(description="ID of the point to update")],
            information: Annotated[str, Field(description="New text content")],
            collection_name: Annotated[
                str, Field(description="The collection containing the point")
            ],
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="New metadata to store with the information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Update an existing point in Qdrant.
            :param ctx: The context for the request.
            :param point_id: The ID of the point to update.
            :param information: The new information to store.
            :param metadata: New JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection containing the point.
            :return: A message indicating the result of the update.
            """
            denied = self._check_access(collection_name, WRITE)
            if denied:
                return denied

            await ctx.debug(f"Updating point {point_id} in Qdrant")

            entry = Entry(content=information, metadata=metadata)
            success = await self.qdrant_connector.update(
                point_id, entry, collection_name=collection_name
            )

            if success:
                if collection_name:
                    return f"Updated point {point_id} in collection {collection_name}"
                return f"Updated point {point_id}"
            else:
                return (
                    f"Failed to update point {point_id}: point or collection not found"
                )

        async def delete(
            ctx: Context,
            point_ids: Annotated[
                list[str], Field(description="List of point IDs to delete")
            ],
            collection_name: Annotated[
                str, Field(description="The collection to delete from")
            ],
        ) -> str:
            """
            Delete points from Qdrant by their IDs.
            :param ctx: The context for the request.
            :param point_ids: The list of point IDs to delete.
            :param collection_name: The name of the collection to delete from.
            :return: A message indicating the result of the deletion.
            """
            denied = self._check_access(collection_name, DELETE)
            if denied:
                return denied

            await ctx.debug(f"Deleting points {point_ids} from Qdrant")

            count = await self.qdrant_connector.delete(
                point_ids, collection_name=collection_name
            )

            if collection_name:
                return f"Deleted {count} point(s) from collection {collection_name}"
            return f"Deleted {count} point(s)"

        async def delete_by_filter(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to delete from")
            ],
            query_filter: ArbitraryFilter,
        ) -> str:
            """
            Delete points from Qdrant that match a filter condition.
            :param ctx: The context for the request.
            :param collection_name: The name of the collection to delete from.
            :param query_filter: The filter to apply for deletion.
            :return: A message indicating the result of the deletion.
            """
            denied = self._check_access(collection_name, DELETE)
            if denied:
                return denied

            await ctx.debug(f"Deleting points by filter from Qdrant: {query_filter}")

            filter_condition = models.Filter(**query_filter) if query_filter else None
            if not filter_condition:
                return "Cannot delete without a filter condition"

            success = await self.qdrant_connector.delete_by_filter(
                filter_condition, collection_name=collection_name
            )

            if success:
                if collection_name:
                    return f"Deleted points matching filter from collection {collection_name}"
                return "Deleted points matching filter"
            else:
                return "Failed to delete points: collection not found"

        find_foo = find
        store_foo = store
        update_foo = update
        delete_foo = delete
        delete_by_filter_foo: Any = delete_by_filter

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})

        # Apply filterable conditions to delete_by_filter if configured
        if len(filterable_conditions) > 0:
            delete_by_filter_foo = wrap_filters(
                delete_by_filter_foo, filterable_conditions
            )
        elif not self.qdrant_settings.allow_arbitrary_filter:
            # If no arbitrary filters allowed, don't register delete_by_filter
            delete_by_filter_foo = None

        # Bind the single served collection (and hide the argument) when exactly one
        # collection is configured; otherwise the argument stays required.
        if self._default_collection:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self._default_collection}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self._default_collection}
            )
            update_foo = make_partial_function(
                update_foo, {"collection_name": self._default_collection}
            )
            delete_foo = make_partial_function(
                delete_foo, {"collection_name": self._default_collection}
            )
            if delete_by_filter_foo is not None:
                delete_by_filter_foo = make_partial_function(
                    delete_by_filter_foo,
                    {"collection_name": self._default_collection},
                )

        # Register tools according to the permissions granted across all served
        # collections (union). In legacy mode this reduces to read + (write/delete
        # unless QDRANT_READ_ONLY).
        if self._expose(READ):
            self._register_tool(
                find_foo,
                name="qdrant-find",
                description=self.tool_settings.tool_find_description,
                perm=READ,
            )

        if self._expose(WRITE):
            self._register_tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
                perm=WRITE,
            )
            self._register_tool(
                update_foo,
                name="qdrant-update",
                description=self.tool_settings.tool_update_description,
                perm=WRITE,
            )

        if self._expose(DELETE):
            self._register_tool(
                delete_foo,
                name="qdrant-delete",
                description=self.tool_settings.tool_delete_description,
                perm=DELETE,
            )
            if delete_by_filter_foo is not None:
                self._register_tool(
                    delete_by_filter_foo,
                    name="qdrant-delete-by-filter",
                    description=self.tool_settings.tool_delete_by_filter_description,
                    perm=DELETE,
                )
