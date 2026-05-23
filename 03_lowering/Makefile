MLIR_OPT ?= mlir-opt
MLIR_TRANSLATE ?= mlir-translate

INPUT = linalg_on_tensors.mlir

STEP1 = linalg_on_buffers.mlir
STEP2 = loops.mlir
STEP3 = host_device_code.mlir
STEP4 = llvm_dialect.mlir

.PHONY: all clean show

all: $(STEP4)

# ------------------------------------------------------------
# Linalg on Tensor -> Linalg on Buffers
#
# tensor<...> becomes memref<...>
# ------------------------------------------------------------
$(STEP1): $(INPUT)
	$(MLIR_OPT) $< \
		-one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
		-o $@

# ------------------------------------------------------------
# Linalg on Buffers -> Loops
#
# linalg.matmul becomes scf.for loops
# ------------------------------------------------------------
$(STEP2): $(STEP1)
	$(MLIR_OPT) $< \
		-convert-linalg-to-loops \
		-o $@

# ------------------------------------------------------------
# Loops -> Host/Device Code
#
# scf.for becomes lower-level control flow
# We still have MLIR, but it is much closer to LLVM lowering.
# ------------------------------------------------------------
$(STEP3): $(STEP2)
	$(MLIR_OPT) $< \
		-convert-scf-to-cf \
		-o $@

# ------------------------------------------------------------
# Host/Device Code -> LLVM Dialect
#
# memref, arith, cf, func are lowered to LLVM dialect.
# ------------------------------------------------------------
$(STEP4): $(STEP3)
	$(MLIR_OPT) $< \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o $@

show:
	@echo "Generated files:"
	@echo "  $(STEP1)  # Linalg on Buffers"
	@echo "  $(STEP2)  # Loops"
	@echo "  $(STEP3)  # Host/Device Code"
	@echo "  $(STEP4)  # LLVM Dialect"

clean:
	rm -f $(STEP1) $(STEP2) $(STEP3) $(STEP4)