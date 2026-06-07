"""Centralized tree-sitter node profiles for supported languages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Literal, Set


ParserMode = Literal["specialized", "generic"]


@dataclass(frozen=True)
class LanguageProfile:
    """Language-specific parser profile."""

    language: str
    tree_sitter_language: str
    parser_mode: ParserMode
    class_node_types: Set[str] = field(default_factory=set)
    method_node_types: Set[str] = field(default_factory=set)
    call_node_types: Set[str] = field(default_factory=set)


def _set(values: Iterable[str]) -> Set[str]:
    return set(values)


LANGUAGE_PROFILES: Dict[str, LanguageProfile] = {
    "bash": LanguageProfile(
        "bash",
        "bash",
        "generic",
        method_node_types=_set({"function_definition"}),
        call_node_types=_set({"command", "command_substitution", "call_expression"}),
    ),
    "shell": LanguageProfile(
        "shell",
        "bash",
        "generic",
        method_node_types=_set({"function_definition"}),
        call_node_types=_set({"command", "command_substitution", "call_expression"}),
    ),
    "python": LanguageProfile("python", "python", "specialized"),
    "java": LanguageProfile("java", "java", "specialized"),
    "javascript": LanguageProfile("javascript", "javascript", "specialized"),
    "typescript": LanguageProfile("typescript", "typescript", "specialized"),
    "tsx": LanguageProfile("tsx", "typescript", "specialized"),
    "c": LanguageProfile("c", "c", "specialized"),
    "cpp": LanguageProfile("cpp", "cpp", "specialized"),
    "csharp": LanguageProfile("csharp", "c_sharp", "specialized"),
    "r": LanguageProfile("r", "r", "specialized"),
    "go": LanguageProfile(
        "go",
        "go",
        "generic",
        method_node_types=_set({"function_declaration", "method_declaration", "function"}),
        call_node_types=_set({"call_expression"}),
    ),
    "coffeescript": LanguageProfile(
        "coffeescript",
        "coffeescript",
        "generic",
        method_node_types=_set({"function", "function_definition"}),
        call_node_types=_set({"call", "call_expression"}),
    ),
    "dart": LanguageProfile(
        "dart",
        "dart",
        "generic",
        class_node_types=_set({"class_definition", "mixin_declaration", "extension_declaration"}),
        method_node_types=_set({"function_signature", "method_signature", "function_body"}),
        call_node_types=_set({"call_expression", "method_invocation"}),
    ),
    "groovy": LanguageProfile(
        "groovy",
        "groovy",
        "generic",
        class_node_types=_set({"class_declaration", "interface_declaration", "trait_declaration"}),
        method_node_types=_set({"method_declaration", "function_declaration"}),
        call_node_types=_set({"method_call", "call_expression"}),
    ),
    "julia": LanguageProfile(
        "julia",
        "julia",
        "generic",
        method_node_types=_set({"function_definition", "short_function_definition"}),
        call_node_types=_set({"call_expression"}),
    ),
    "kotlin": LanguageProfile(
        "kotlin",
        "kotlin",
        "generic",
        class_node_types=_set({"class_declaration", "object_declaration", "interface_declaration"}),
        method_node_types=_set({"function_declaration"}),
        call_node_types=_set({"call_expression"}),
    ),
    "lisp": LanguageProfile(
        "lisp",
        "commonlisp",
        "generic",
        method_node_types=_set({"function_definition", "defun"}),
        call_node_types=_set({"list_lit", "call", "call_expression"}),
    ),
    "lua": LanguageProfile(
        "lua",
        "lua",
        "generic",
        method_node_types=_set({"function_definition", "function_declaration"}),
        call_node_types=_set({"function_call", "call_expression"}),
    ),
    "objectivec": LanguageProfile(
        "objectivec",
        "objc",
        "generic",
        class_node_types=_set({"class_interface", "class_implementation", "protocol_declaration"}),
        method_node_types=_set({"function_definition", "method_definition"}),
        call_node_types=_set({"call_expression", "message_expression"}),
    ),
    "perl": LanguageProfile(
        "perl",
        "perl",
        "generic",
        method_node_types=_set({"subroutine_definition", "function_definition"}),
        call_node_types=_set({"call_expression", "method_call"}),
    ),
    "php": LanguageProfile(
        "php",
        "php",
        "generic",
        class_node_types=_set({"class_declaration", "interface_declaration", "trait_declaration"}),
        method_node_types=_set({"function_definition", "method_declaration"}),
        call_node_types=_set({"function_call_expression", "call_expression"}),
    ),
    "ruby": LanguageProfile(
        "ruby",
        "ruby",
        "generic",
        class_node_types=_set({"class", "module"}),
        method_node_types=_set({"method", "singleton_method"}),
        call_node_types=_set({"call", "command", "method_call"}),
    ),
    "rust": LanguageProfile(
        "rust",
        "rust",
        "specialized",
        class_node_types=_set({"struct_item", "enum_item", "impl_item", "trait_item"}),
        method_node_types=_set({"function_item"}),
        call_node_types=_set({"call_expression", "method_call_expression"}),
    ),
    "swift": LanguageProfile(
        "swift",
        "swift",
        "generic",
        class_node_types=_set(
            {"class_declaration", "struct_declaration", "protocol_declaration", "extension_declaration"}
        ),
        method_node_types=_set({"function_declaration", "initializer_declaration", "deinitializer_declaration"}),
        call_node_types=_set({"call_expression", "function_call_expression"}),
    ),
    "haskell": LanguageProfile(
        "haskell",
        "haskell",
        "generic",
        class_node_types=_set({"class_declaration", "instance_declaration", "data_declaration"}),
        method_node_types=_set({"function_declaration", "function"}),
        call_node_types=_set({"apply", "call_expression", "call"}),
    ),
    "tcl": LanguageProfile(
        "tcl",
        "tcl",
        "generic",
        method_node_types=_set({"proc_definition", "function_definition"}),
        call_node_types=_set({"command", "call_expression"}),
    ),
    "zig": LanguageProfile(
        "zig",
        "zig",
        "generic",
        class_node_types=_set({"struct_declaration", "union_declaration", "enum_declaration"}),
        method_node_types=_set({"function_declaration"}),
        call_node_types=_set({"call_expression"}),
    ),
    "ocaml": LanguageProfile(
        "ocaml",
        "ocaml",
        "generic",
        class_node_types=_set({"class_definition", "module_definition"}),
        method_node_types=_set({"function_definition", "value_definition", "let_binding"}),
        call_node_types=_set({"call_expression", "call", "apply"}),
    ),
    "cmake": LanguageProfile(
        "cmake",
        "cmake",
        "generic",
        method_node_types=_set({"function_def", "macro_def", "function_definition"}),
        call_node_types=_set({"normal_command", "call_expression", "call"}),
    ),
    "css": LanguageProfile(
        "css",
        "css",
        "generic",
        class_node_types=_set({"rule_set", "at_rule"}),
        method_node_types=_set({"rule_set", "at_rule"}),
        call_node_types=_set({"function_call", "call_expression", "call"}),
    ),
    "yaml": LanguageProfile(
        "yaml",
        "yaml",
        "generic",
        method_node_types=_set({"block_mapping_pair", "flow_pair"}),
        call_node_types=_set({"flow_node", "plain_scalar", "call_expression", "call"}),
    ),
    "xml": LanguageProfile(
        "xml",
        "xml",
        "generic",
        method_node_types=_set({"element", "empty_elem_tag"}),
        call_node_types=_set({"start_tag", "empty_elem_tag", "call_expression", "call"}),
    ),
}
