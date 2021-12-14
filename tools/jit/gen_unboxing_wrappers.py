# Generates CodegenUnboxingWrappers.cpp.
# This generates static unboxing wrapper for ATen ops.
import json
from dataclasses import dataclass
from tools.codegen.api.types import CppSignatureGroup
from tools.codegen.context import method_with_native_function
from tools.codegen.model import NativeFunction
from typing import Optional


@dataclass(frozen=True)
class ComputeUnboxingWrapper:

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        # We unconditionally generate function wrappers,
        sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding)

        sig = sig_group.signature

        # escape double quote in schema, get rid of extra double quotes
        schema = json.dumps(sig.func.__str__())[1:-1]

        return f"""
OperatorGenerator(
    TORCH_SELECTIVE_SCHEMA("aten::{schema}"),
    [](Stack & stack) {{
        RECORD_FUNCTION("{sig.name()}", std::vector<c10::IValue>());
        at::unboxing::{f.func.name.unambiguous_name()}(stack);
    }},
    aliasAnalysisFromSchema()
),
"""
