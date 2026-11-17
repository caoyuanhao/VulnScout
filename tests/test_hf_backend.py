from vulnscout.hf_backend import parse_tool_calls


def test_parse_tool_calls():
    text = ('Let me check.\n<tool_call>\n{"name": "search_cpe", "arguments": {"keyword": "log4j"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "check_package", "arguments": "{\\"ecosystem\\": \\"Maven\\"}"}\n</tool_call>')
    content, calls = parse_tool_calls(text, 3)
    assert content == "Let me check."
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("call_3_0", "search_cpe", {"keyword": "log4j"}),
        ("call_3_1", "check_package", {"ecosystem": "Maven"}),     # string arguments are decoded
    ]


def test_parse_malformed_call_is_reported():
    _, calls = parse_tool_calls('<tool_call>{"name": "search_cpe", "arguments": {"keyword": }</tool_call>', 1)
    assert calls[0].name == "__invalid__"


def test_plain_answer():
    assert parse_tool_calls("No CVEs found.", 1) == ("No CVEs found.", [])
