from app.pii import mask_pii
from app.chunking import chunk_text
from app.store import VectorStore


def test_pii_masking():
    t, c = mask_pii("Mail a.b@x.com or call +91 9876543210, PAN ABCDE1234F, Aadhaar 1234 5678 9012")
    assert "[EMAIL]" in t and "[PHONE]" in t and "[PAN]" in t and "[AADHAAR]" in t
    assert "9876543210" not in t and c["EMAIL"] == 1


def test_chunking_overlap_and_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    chunks = chunk_text(text, 700, 20)
    assert len(chunks) > 1 and all(len(c) < 1500 for c in chunks)


def test_retrieval_relevance():
    s = VectorStore()
    s.add(["Employees get 24 days of paid leave per year."], "hr.txt", "t1")
    s.add(["Laptops must use disk encryption and strong passwords."], "sec.txt", "t1")
    assert s.search("how many leave days", "t1")[0]["source"] == "hr.txt"


def test_tenant_isolation():
    s = VectorStore()
    s.add(["Secret salary data for tenant A."], "a.txt", "A")
    assert s.search("salary", "B") == []
