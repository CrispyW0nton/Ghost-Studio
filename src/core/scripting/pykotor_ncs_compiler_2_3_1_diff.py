# SPDX-License-Identifier: LGPL-3.0-or-later
"""Patch to PyKotor 2.3.1 ``pykotor/resource/formats/ncs/compiler/classes.py``.

PyKotor (https://github.com/OldRepublicDevs/PyKotor) is licensed under the GNU
Lesser General Public License v3.0 or later. This module holds only a unified
diff against that file, so it is distributed under the same licence; the rest
of Ghost Studio stays MIT. ``pykotor_ncs_compiler_fix`` applies it in memory
and never modifies the installed package.

What the patch fixes (found while repairing KOTOR II ``k_ai_master.ncs``):

* Forward-declared functions were linked by *value* lookup of their NOP label,
  so equal NOPs sent most JSR calls of a large script to one wrong body.
  Labels are now found by identity, and called prototypes without bodies are
  rejected.
* ACTION (engine call) arguments are pushed last-first, because retail handlers
  pop the first declared parameter first.
* ``switch`` selectors stay in lexical scope until a case exits; nested local
  reads, return cleanup and saved actions account for them, and ``default`` is
  dispatched after the case tests.
* The entry wrapper matches retail: a ``JSR; RETN`` globals stub, then
  ``SAVEBP; JSR main; RESTOREBP; MOVSP; RETN``.
"""

SOURCE_SHA256 = "3f732f252dae5d03cbc581759d46e07dbfb8fd6c0d3207f9de9951416207feff"
PATCHED_SHA256 = "38346a654a49197db2eccef939e34a7c58f650c73ccabbc08f67300fb36912f0"

DIFF = r'''--- a/pykotor/resource/formats/ncs/compiler/classes.py
+++ b/pykotor/resource/formats/ncs/compiler/classes.py
@@ -369,6 +369,8 @@
             self.library_lookup = normalized
 
         self.function_map: dict[str, FunctionReference] = {}
+        self.defined_functions: set[str] = set()
+        self.referenced_functions: set[str] = set()
         self._global_scope: list[ScopedValue] = []
         self.struct_map: dict[str, Struct] = {}
 
@@ -394,33 +396,49 @@
         ]
         others: list[TopLevelObject] = [obj for obj in self.objects if obj not in included and obj not in script_globals]
 
-        if script_globals:
-            for global_def in script_globals:
-                global_def.compile(ncs, self)
-            ncs.add(NCSInstructionType.SAVEBP, args=[])
-        entry_index: int = len(ncs.instructions)
+        # Compile globals first so struct layouts and their stack sizes are known.
+        for global_def in script_globals:
+            global_def.compile(ncs, self)
+        global_size = -self.scope_size()
+        if global_size:
+            ncs.add(NCSInstructionType.SAVEBP)
+        entry_index = len(ncs.instructions)
 
         for obj in others:
             obj.compile(ncs, self)
 
-        if "main" in self.function_map:
-            ncs.add(NCSInstructionType.RETN, args=[], index=entry_index)
-            ncs.add(
-                NCSInstructionType.JSR,
-                jump=self.function_map["main"][0],
-                index=entry_index,
-            )
-        elif "StartingConditional" in self.function_map:
-            ncs.add(NCSInstructionType.RETN, args=[], index=entry_index)
-            ncs.add(
-                NCSInstructionType.JSR,
-                jump=self.function_map["StartingConditional"][0],
-                index=entry_index,
-            )
-            ncs.add(NCSInstructionType.RSADDI, args=[], index=entry_index)
-        else:
-            msg = "This file has no entry point and cannot be compiled (Most likely an include file)."
-            raise EntryPointError(msg)
+        entry_name = "main" if "main" in self.defined_functions else "StartingConditional"
+        if entry_name not in self.defined_functions:
+            raise EntryPointError("This file has no entry point with a body.")
+        unresolved = self.referenced_functions - self.defined_functions
+        if unresolved:
+            raise CompileError("Calls to prototypes without bodies: " + ", ".join(sorted(unresolved)))
+
+        target = self.function_map[entry_name].instruction
+        conditional = entry_name == "StartingConditional"
+        wrapper = NCS()
+        if conditional:
+            wrapper.add(NCSInstructionType.RSADDI)
+        wrapper.add(NCSInstructionType.JSR, jump=target)
+        if global_size:
+            if conditional:
+                # Preserve the conditional result across RESTOREBP and globals cleanup.
+                wrapper.add(NCSInstructionType.CPDOWNSP, args=[-global_size - 12, 4])
+                wrapper.add(NCSInstructionType.MOVSP, args=[-4])
+            wrapper.add(NCSInstructionType.RESTOREBP)
+            wrapper.add(NCSInstructionType.MOVSP, args=[-global_size])
+        wrapper.add(NCSInstructionType.RETN)
+        ncs.instructions[entry_index:entry_index] = wrapper.instructions
+
+        if global_size:
+            # Retail enters global initialization via a JSR/RETN stub.
+            globals_entry = ncs.instructions[0]
+            stub = NCS()
+            if conditional:
+                stub.add(NCSInstructionType.RSADDI)
+            stub.add(NCSInstructionType.JSR, jump=globals_entry)
+            stub.add(NCSInstructionType.RETN)
+            ncs.instructions[:0] = stub.instructions
 
     def compile_jsr(
         self,
@@ -432,6 +450,7 @@
         args_list = list(args)
 
         func_map: FunctionReference = self.function_map[name]
+        self.referenced_functions.add(name)
         definition: FunctionForwardDeclaration | FunctionDefinition = func_map.definition
         start_instruction: NCSInstruction = func_map.instruction
 
@@ -558,7 +577,9 @@
         self.scope: list[ScopedValue] = []
         self._parent: CodeBlock | None = None
         self._statements: list[Statement] = []
-        self._break_scope: bool = False
+        self._break_scope = False
+        self._continue_scope = False
+        self.fixed_stack_size = 0
         self.temp_stack: int = 0
 
     def add(self, statement: Statement):
@@ -579,37 +600,8 @@
         self.temp_stack = 0
 
         for statement in self._statements:
-            if not isinstance(statement, ReturnStatement):
-                statement.compile(
-                    ncs,
-                    root,
-                    self,
-                    return_instruction,
-                    break_instruction,
-                    continue_instruction,
-                )
-            else:
-                scope_size = self.full_scope_size(root)
-
-                return_type: DynamicDataType = statement.compile(
-                    ncs,
-                    root,
-                    self,
-                    return_instruction,
-                    break_instruction=None,
-                    continue_instruction=None,
-                )
-                if return_type != DynamicDataType.VOID:
-                    ncs.add(
-                        NCSInstructionType.CPDOWNSP,
-                        args=[-scope_size - return_type.size(root) * 2, 4],
-                    )
-                    ncs.add(NCSInstructionType.MOVSP, args=[-return_type.size(root)])
-
-                # External compiler optimizes away MOVSP with offset 0, so we should match that behavior
-                if scope_size != 0:
-                    ncs.add(NCSInstructionType.MOVSP, args=[-scope_size])
-                ncs.add(NCSInstructionType.JMP, jump=return_instruction)
+            statement.compile(ncs, root, self, return_instruction, break_instruction, continue_instruction)
+            if isinstance(statement, ReturnStatement):
                 return
         # External compiler optimizes away MOVSP with offset 0, so we should match that behavior
         scope_size = self.scope_size(root)
@@ -644,13 +636,13 @@
                 break
         else:
             if self._parent is not None:
-                return self._parent.get_scoped(identifier, root, offset)
+                return self._parent.get_scoped(identifier, root, offset - self.fixed_stack_size)
             return root.get_scoped(identifier, root)
         return GetScopedResult(is_global=False, datatype=scoped.data_type, offset=offset, is_const=scoped.is_const)
 
     def scope_size(self, root: CodeRoot) -> int:
         """Returns size of local scope."""
-        return sum(scoped.data_type.size(root) for scoped in self.scope)
+        return self.fixed_stack_size + sum(scoped.data_type.size(root) for scoped in self.scope)
 
     def full_scope_size(self, root: CodeRoot) -> int:
         """Returns size of scope, including outer blocks."""
@@ -660,16 +652,19 @@
             size += self._parent.full_scope_size(root)
         return size
 
-    def break_scope_size(self, root: CodeRoot) -> int:
-        """Returns size of scope up to the nearest loop/switch statement."""
-        size = 0
-        size += self.scope_size(root)
-        if self._parent is not None and not self._parent._break_scope:  # noqa: SLF001
-            size += self._parent.break_scope_size(root)
+    def full_temp_size(self) -> int:
+        return self.temp_stack + (self._parent.full_temp_size() if self._parent is not None else 0)
+
+    def control_scope_size(self, root: CodeRoot, *, continuing: bool = False) -> int:
+        size = self.scope_size(root)
+        boundary = self._continue_scope if continuing else self._break_scope
+        if self._parent is not None and not boundary:
+            size += self._parent.control_scope_size(root, continuing=continuing)
         return size
 
-    def mark_break_scope(self):
+    def mark_control_scope(self, *, loop: bool = False):
         self._break_scope = True
+        self._continue_scope = loop
 
 
 class ScopedValue:
@@ -748,7 +743,7 @@
                 msg = f"Non-constant default value specified for function prototype parameter '{param.identifier}'."
                 raise CompileError(msg)
 
-        if name in root.function_map and not root.function_map[name].is_prototype():
+        if name in root.defined_functions:
             msg = f"Function '{name}' is already defined\n  Cannot redefine a function that already has an implementation"
             raise CompileError(msg)
         if name in root.function_map and root.function_map[name].is_prototype():
@@ -761,6 +756,7 @@
             ncs.instructions.append(retn)
 
             root.function_map[name] = FunctionReference(function_start, self)
+        root.defined_functions.add(name)
 
     def _compile_function(self, root: CodeRoot, name: str, ncs: NCS):  # noqa: D417
         if not self.is_matching_signature(root.function_map[name].definition):
@@ -787,7 +783,9 @@
         self.block.compile(temp, root, None, retn, None, None)
         temp.instructions.append(retn)
 
-        stub_index: int = ncs.instructions.index(root.function_map[name].instruction)
+        # NCSInstruction equality is structural. Distinct NOP labels compare equal.
+        stub = root.function_map[name].instruction
+        stub_index = next(i for i, ins in enumerate(ncs.instructions) if ins is stub)
         ncs.instructions[stub_index + 1 : stub_index + 1] = temp.instructions
 
     def is_matching_signature(self, prototype: FunctionForwardDeclaration | FunctionDefinition) -> bool:
@@ -1286,29 +1284,25 @@
                 elif constant.datatype == DataType.OBJECT:
                     self._args.append(ObjectExpression(int(constant.value)))
         this_stack = 0
-        # DEBUG: Log arguments before compilation
-
-        # Compile arguments in FORWARD order (left to right, first argument first)
-        # NCS bytecode pushes arguments left-to-right, so when the interpreter
-        # pops them (last-in-first-out), args_snap has them in reverse order.
-        # The interpreter then reverses args_snap to match function.params order.
-        # - Compile in forward: push fFloat (first param, bottom), push nWidth, push nDecimals (last param, top)
-        # - Stack: [fFloat, nWidth, nDecimals] (nDecimals on top)
-        # - Interpreter pops: nDecimals -> args_snap[0], nWidth -> args_snap[1], fFloat -> args_snap[2]
-        # - Before reverse: args_snap = [nDecimals, nWidth, fFloat]
-        # - After reverse: args_snap = [fFloat, nWidth, nDecimals] - CORRECT!
-        for i in range(len(self._args)):  # Iterate in forward order
-            arg = self._args[i]
-            param_index = i  # Parameter index in forward order
+        # Retail ACTION pops the first declared argument first.
+        # Evaluate engine arguments right-to-left so it is at the top of the stack.
+        for param_index in reversed(range(len(self._args))):
+            arg = self._args[param_index]
             param_type = DynamicDataType(self._function.params[param_index].datatype)
             if param_type == DataType.ACTION:
                 after_command = NCSInstruction()
                 ncs.add(
                     NCSInstructionType.STORE_STATE,
-                    args=[-root.scope_size(), block.full_scope_size(root)],
+                    args=[-root.scope_size(), block.full_scope_size(root) + block.full_temp_size()],
                 )
                 ncs.add(NCSInstructionType.JMP, jump=after_command)
-                arg.compile(ncs, root, block)
+                # The deferred expression runs on a separate saved stack.
+                # Discard its return value there, and restore the caller's model.
+                caller_temps = block.temp_stack
+                action_result = arg.compile(ncs, root, block)
+                if action_result != DynamicDataType.VOID:
+                    ncs.add(NCSInstructionType.MOVSP, args=[-action_result.size(root)])
+                block.temp_stack = caller_temps
                 ncs.add(NCSInstructionType.RETN)
 
                 ncs.instructions.append(after_command)
@@ -2539,8 +2533,16 @@
         break_instruction: NCSInstruction | None,
         continue_instruction: NCSInstruction | None,
     ) -> DynamicDataType:
+        scope_size = block.full_scope_size(root)
         if self.expression is not None:
-            return self.expression.compile(ncs, root, block)
+            return_type = self.expression.compile(ncs, root, block)
+            size = return_type.size(root)
+            if size:
+                ncs.add(NCSInstructionType.CPDOWNSP, args=[-scope_size - size * 2, size])
+                ncs.add(NCSInstructionType.MOVSP, args=[-size])
+        if scope_size:
+            ncs.add(NCSInstructionType.MOVSP, args=[-scope_size])
+        ncs.add(NCSInstructionType.JMP, jump=return_instruction)
         return DynamicDataType.VOID
 
 
@@ -2559,8 +2561,7 @@
         break_instruction: NCSInstruction | None,
         continue_instruction: NCSInstruction | None,
     ):
-        # Tell break/continue statements to stop here when getting scope size
-        block.mark_break_scope()
+        self.block.mark_control_scope(loop=True)
 
         loopstart = ncs.add(NCSInstructionType.NOP, args=[])
         loopend = NCSInstruction(NCSInstructionType.NOP, args=[])
@@ -2599,8 +2600,7 @@
         break_instruction: NCSInstruction | None,
         continue_instruction: NCSInstruction | None,
     ):
-        # Tell break/continue statements to stop here when getting scope size
-        block.mark_break_scope()
+        self.block.mark_control_scope(loop=True)
 
         loopstart = ncs.add(NCSInstructionType.NOP, args=[])
         conditionstart = NCSInstruction(NCSInstructionType.NOP, args=[])
@@ -2656,8 +2656,7 @@
         break_instruction: NCSInstruction | None,
         continue_instruction: NCSInstruction | None,
     ):
-        # Tell break/continue statements to stop here when getting scope size
-        block.mark_break_scope()
+        self.block.mark_control_scope(loop=True)
 
         if self.initial is not None:
             if isinstance(self.initial, Statement):
@@ -2751,7 +2750,7 @@
         if break_instruction is None:
             msg = "break statement not inside loop or switch\n  break can only be used inside while, do-while, for, or switch statements"
             raise CompileError(msg)
-        ncs.add(NCSInstructionType.MOVSP, args=[-block.break_scope_size(root)])
+        ncs.add(NCSInstructionType.MOVSP, args=[-block.control_scope_size(root)])
         ncs.add(NCSInstructionType.JMP, jump=break_instruction)
 
 
@@ -2771,7 +2770,7 @@
         if continue_instruction is None:
             msg = "continue statement not inside loop\n  continue can only be used inside while, do-while, or for loops"
             raise CompileError(msg)
-        ncs.add(NCSInstructionType.MOVSP, args=[-block.break_scope_size(root)])
+        ncs.add(NCSInstructionType.MOVSP, args=[-block.control_scope_size(root, continuing=True)])
         ncs.add(NCSInstructionType.JMP, jump=continue_instruction)
 
 
@@ -2902,8 +2901,6 @@
         self.expression: Expression = expression
         self.blocks: list[SwitchBlock] = blocks
 
-        self.real_block: CodeBlock = CodeBlock()
-
     def compile(
         self,
         ncs: NCS,
@@ -2913,54 +2910,43 @@
         break_instruction: NCSInstruction | None,
         continue_instruction: NCSInstruction | None,
     ):
-        self.real_block._parent = block  # noqa: SLF001
-        block.mark_break_scope()
-
-        block = self.real_block
-
+        before_selector = block.temp_stack
         expression_type = self.expression.compile(ncs, root, block)
-        block.temp_stack += expression_type.size(root)
-
-        end_of_switch = NCSInstruction(NCSInstructionType.NOP, args=[])
-
-        tempncs = NCS()
-        switchblock_to_instruction = {}
-        for switchblock in self.blocks:
-            switchblock_start = tempncs.add(NCSInstructionType.NOP, args=[])
-            switchblock_to_instruction[switchblock] = switchblock_start
-            for statement in switchblock.block:
-                statement.compile(
-                    tempncs,
-                    root,
-                    block,
-                    return_instruction,
-                    end_of_switch,
-                    None,
-                )
-
-        for switchblock in self.blocks:
-            for label in switchblock.labels:
-                # Do not want to run switch expression multiple times, execute it once and copy it to the top
-                ncs.add(
-                    NCSInstructionType.CPTOPSP,
-                    args=[-expression_type.size(root), expression_type.size(root)],
-                )
-                label.compile(
-                    ncs,
-                    root,
-                    block,
-                    switchblock_to_instruction[switchblock],
-                    expression_type,
-                )
-        # If none of the labels match, jump over the code block
-        ncs.add(NCSInstructionType.JMP, jump=end_of_switch)
-
-        ncs.merge(tempncs)
-        ncs.instructions.append(end_of_switch)
-
-        # Pop the Switch expression
+        block.temp_stack = before_selector
+        if expression_type != DynamicDataType.INT:
+            raise CompileError("Switch selector must be an integer")
+
+        # The selector stays live for all case bodies and is underneath case locals.
+        selector_scope = CodeBlock()
+        selector_scope._parent = block
+        selector_scope.fixed_stack_size = expression_type.size(root)
+        end = NCSInstruction(NCSInstructionType.NOP)
+        entries = [NCSInstruction(NCSInstructionType.NOP) for _ in self.blocks]
+        default = end
+
+        for entry, case in zip(entries, self.blocks):
+            for label in case.labels:
+                if isinstance(label, DefaultSwitchLabel):
+                    if default is not end:
+                        raise CompileError("Duplicate default switch label")
+                    default = entry
+                    continue
+                ncs.add(NCSInstructionType.CPTOPSP, args=[-4, 4])
+                selector_scope.temp_stack = 4
+                label.compile(ncs, root, selector_scope, entry, expression_type)
+                selector_scope.temp_stack = 0
+        ncs.add(NCSInstructionType.JMP, jump=default)
+
+        for entry, case in zip(entries, self.blocks):
+            ncs.instructions.append(entry)
+            case_scope = CodeBlock()
+            case_scope.mark_control_scope()
+            for statement in case.block:
+                case_scope.add(statement)
+            case_scope.compile(ncs, root, selector_scope, return_instruction, end, continue_instruction)
+
+        ncs.instructions.append(end)
         ncs.add(NCSInstructionType.MOVSP, args=[-4])
-        block.temp_stack -= expression_type.size(root)
 
 
 class SwitchBlock:
'''
