import unittest

from torchgen.context import native_function_manager
from torchgen.gen_functionalization_type import wrap_propagate_mutations_and_return
from torchgen.model import Location, NativeFunction


class TestFunctionalizationCodegen(unittest.TestCase):
    def test_mutations_do_not_require_removed_xla_runtime(self) -> None:
        cases = (
            (
                "op_(Tensor(a!) self) -> Tensor(a!)",
                "op(Tensor self) -> Tensor",
                ("self",),
                "return self;",
            ),
            (
                "op.out(Tensor self, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))",
                "op(Tensor self) -> (Tensor, Tensor)",
                ("out0", "out1"),
                "out0, out1",
            ),
        )
        for schema, functional_schema, mutable_args, returned in cases:
            with self.subTest(schema=schema):
                f, _ = NativeFunction.from_yaml(
                    {"func": schema}, Location(__file__, 1), valid_tags=set()
                )
                functional_op, _ = NativeFunction.from_yaml(
                    {"func": functional_schema},
                    Location(__file__, 1),
                    valid_tags=set(),
                )
                with native_function_manager(f):
                    code = wrap_propagate_mutations_and_return(
                        f, functional_op, "output"
                    )

                self.assertNotIn("propagate_xla_data_direct", code)
                self.assertNotIn("from_functional_tensor", code)
                for i, arg in enumerate(mutable_args):
                    result = (
                        "output"
                        if len(mutable_args) == 1
                        else f"std::get<{i}>(output)"
                    )
                    self.assertIn(
                        f"replace_({arg}, {result});\n"
                        f"  at::functionalization::impl::commit_update({arg});\n"
                        f"  at::functionalization::impl::sync({arg});",
                        code,
                    )
                self.assertIn(returned, code)


if __name__ == "__main__":
    unittest.main()
