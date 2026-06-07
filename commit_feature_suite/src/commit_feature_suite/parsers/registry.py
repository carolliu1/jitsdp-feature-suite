"""Parser registry with class mapping + generic fallback."""

from __future__ import annotations

from typing import Dict, Type

from commit_feature_suite.parsers.bash_parser import BashParser
from commit_feature_suite.parsers.c_parser import CParser
from commit_feature_suite.parsers.cmake_parser import CMakeParser
from commit_feature_suite.parsers.coffeescript_parser import CoffeeScriptParser
from commit_feature_suite.parsers.cpp_parser import CppParser
from commit_feature_suite.parsers.csharp_parser import CSharpParser
from commit_feature_suite.parsers.css_parser import CssParser
from commit_feature_suite.parsers.dart_parser import DartParser
from commit_feature_suite.parsers.generic_parser import GenericTreeSitterParser
from commit_feature_suite.parsers.go_parser import GoParser
from commit_feature_suite.parsers.groovy_parser import GroovyParser
from commit_feature_suite.parsers.haskell_parser import HaskellParser
from commit_feature_suite.parsers.java_parser import JavaParser
from commit_feature_suite.parsers.javascript_parser import JavaScriptParser
from commit_feature_suite.parsers.julia_parser import JuliaParser
from commit_feature_suite.parsers.kotlin_parser import KotlinParser
from commit_feature_suite.parsers.language_profiles import LANGUAGE_PROFILES
from commit_feature_suite.parsers.lisp_parser import LispParser
from commit_feature_suite.parsers.lua_parser import LuaParser
from commit_feature_suite.parsers.ocaml_parser import OCamlParser
from commit_feature_suite.parsers.objectivec_parser import ObjectiveCParser
from commit_feature_suite.parsers.perl_parser import PerlParser
from commit_feature_suite.parsers.php_parser import PhpParser
from commit_feature_suite.parsers.python_parser import PythonParser
from commit_feature_suite.parsers.r_parser import RParser
from commit_feature_suite.parsers.ruby_parser import RubyParser
from commit_feature_suite.parsers.rust_parser import RustParser
from commit_feature_suite.parsers.shell_parser import ShellParser
from commit_feature_suite.parsers.swift_parser import SwiftParser
from commit_feature_suite.parsers.tcl_parser import TclParser
from commit_feature_suite.parsers.typescript_parser import TypeScriptParser
from commit_feature_suite.parsers.xml_parser import XmlParser
from commit_feature_suite.parsers.yaml_parser import YamlParser
from commit_feature_suite.parsers.zig_parser import ZigParser


PARSER_REGISTRY: Dict[str, Type] = {
    "bash": BashParser,
    "shell": ShellParser,
    "python": PythonParser,
    "java": JavaParser,
    "javascript": JavaScriptParser,
    "typescript": TypeScriptParser,
    "tsx": TypeScriptParser,
    "c": CParser,
    "coffeescript": CoffeeScriptParser,
    "cpp": CppParser,
    "csharp": CSharpParser,
    "dart": DartParser,
    "r": RParser,
    "go": GoParser,
    "groovy": GroovyParser,
    "julia": JuliaParser,
    "kotlin": KotlinParser,
    "lisp": LispParser,
    "lua": LuaParser,
    "objectivec": ObjectiveCParser,
    "perl": PerlParser,
    "php": PhpParser,
    "ruby": RubyParser,
    "rust": RustParser,
    "swift": SwiftParser,
    "tcl": TclParser,
    "haskell": HaskellParser,
    "ocaml": OCamlParser,
    "zig": ZigParser,
    "cmake": CMakeParser,
    "css": CssParser,
    "yaml": YamlParser,
    "xml": XmlParser,
}

SUPPORTED_TREE_SITTER_LANGUAGES = set(LANGUAGE_PROFILES.keys())


def create_language_parser(language: str):
    """Return a parser instance for the given language."""
    normalized = language.lower()
    parser_cls = PARSER_REGISTRY.get(normalized)
    if parser_cls is not None:
        return parser_cls()

    profile = LANGUAGE_PROFILES.get(normalized)
    if profile is not None:
        return GenericTreeSitterParser(profile=profile)
    return GenericTreeSitterParser(normalized)


def has_tree_sitter_parser(language: str) -> bool:
    """Whether tree-sitter-based extraction is configured for this language."""
    return language.lower() in SUPPORTED_TREE_SITTER_LANGUAGES

