"""CWE / ATT&CK parsing and retrieval, on small hand-written catalogs (no download)."""
import pytest

from vulnscout import knowledge

CWE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<Weakness_Catalog Name="CWE" Version="5.0" xmlns="http://cwe.mitre.org/cwe-8">
 <Weaknesses>
  <Weakness ID="89" Name="Improper Neutralization of Special Elements used in an SQL Command" Abstraction="Base" Status="Stable">
   <Description>The product constructs an SQL command using externally-influenced input.</Description>
   <Related_Weaknesses><Related_Weakness Nature="ChildOf" CWE_ID="943" View_ID="1000"/></Related_Weaknesses>
   <Alternate_Terms><Alternate_Term><Term>SQL Injection</Term></Alternate_Term><Alternate_Term><Term>SQLi</Term></Alternate_Term></Alternate_Terms>
   <Common_Consequences><Consequence><Scope>Confidentiality</Scope><Impact>Read Application Data</Impact></Consequence></Common_Consequences>
   <Potential_Mitigations><Mitigation><Phase>Implementation</Phase><Description>Use <b>parameterized</b> queries.</Description></Mitigation></Potential_Mitigations>
   <Observed_Examples><Observed_Example><Reference>CVE-2021-0001</Reference><Description>SQL injection in a login form.</Description></Observed_Example></Observed_Examples>
  </Weakness>
  <Weakness ID="502" Name="Deserialization of Untrusted Data" Abstraction="Base" Status="Draft">
   <Description>The product deserializes untrusted data without verifying that the resulting data will be valid.</Description>
  </Weakness>
  <Weakness ID="1" Name="Old Entry" Abstraction="Base" Status="Deprecated"><Description>gone</Description></Weakness>
 </Weaknesses>
</Weakness_Catalog>"""

ATTACK = {"objects": [
    {"type": "attack-pattern", "id": "attack-pattern--1", "name": "Exploit Public-Facing Application",
     "description": "Adversaries may exploit a weakness in an Internet-facing host (Citation: X). See [ref](https://x).",
     "kill_chain_phases": [{"phase_name": "initial-access"}], "x_mitre_platforms": ["Linux"],
     "external_references": [{"source_name": "mitre-attack", "external_id": "T1190",
                              "url": "https://attack.mitre.org/techniques/T1190"}]},
    {"type": "attack-pattern", "id": "attack-pattern--2", "name": "Revoked Thing", "revoked": True,
     "external_references": [{"source_name": "mitre-attack", "external_id": "T9999", "url": "u"}]},
    {"type": "course-of-action", "id": "course-of-action--1", "name": "Update Software",
     "external_references": [{"source_name": "mitre-attack", "external_id": "M1051", "url": "u"}]},
    {"type": "relationship", "relationship_type": "mitigates", "source_ref": "course-of-action--1",
     "target_ref": "attack-pattern--1"},
]}


@pytest.fixture
def index(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, "DATA", tmp_path)
    monkeypatch.setattr(knowledge, "INDEX", tmp_path / "index.json")
    entries = knowledge.build(CWE_XML, ATTACK)
    yield entries
    knowledge.load.cache_clear()


def test_parse(index):
    ids = [e["id"] for e in index]
    assert ids == ["CWE-89", "CWE-502", "T1190"]          # deprecated and revoked entries dropped
    cwe = index[0]
    assert cwe["terms"] == ["SQL Injection", "SQLi"] and cwe["parents"] == ["CWE-943"]
    assert cwe["mitigations"] == ["[Implementation] Use parameterized queries."]
    assert cwe["examples"][0]["cve"] == "CVE-2021-0001"
    assert cwe["url"] == "https://cwe.mitre.org/data/definitions/89.html"
    t = index[2]
    assert t["description"] == "Adversaries may exploit a weakness in an Internet-facing host . See ref."
    assert t["mitigations"] == ["M1051 Update Software"] and t["tactics"] == ["initial-access"]


def test_search(index):
    assert knowledge.search("what is SQLi?")["results"][0]["id"] == "CWE-89"     # alternate term
    assert knowledge.search("cwe-502")["results"][0]["id"] == "CWE-502"          # exact ID, any case
    assert knowledge.search("internet-facing exploit", source="attack")["results"][0]["id"] == "T1190"
    assert [e["id"] for e in knowledge.search("SQL", source="attack")["results"]] == []
    assert knowledge.search("quantum teleportation")["results"] == []
    full, = [e for e in knowledge.search("CWE-89")["results"]]
    assert "mitigations" in full                                                  # best match in full
