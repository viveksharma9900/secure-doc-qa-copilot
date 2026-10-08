"""Semantic Kernel plugins: retrieval and privacy exposed as kernel functions.
Register with: kernel.add_plugin(RetrievalPlugin(store), plugin_name="retrieval")"""
from semantic_kernel.functions import kernel_function
from .pii import mask_pii


class PrivacyPlugin:
    @kernel_function(name="mask_pii", description="Mask emails, phones, Aadhaar, PAN and card numbers in text.")
    def mask(self, text: str) -> str:
        return mask_pii(text)[0]


class RetrievalPlugin:
    def __init__(self, store, tenant: str = "demo"):
        self.store, self.tenant = store, tenant

    @kernel_function(name="search", description="Search uploaded documents for passages relevant to a query.")
    def search(self, query: str) -> str:
        return "\n".join(f"[{h['source']}] {h['text']}" for h in self.store.search(query, self.tenant))
