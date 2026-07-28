# Makefile for CH32V305 Dual USB (CDC + HID) firmware
# Toolchain: riscv-none-embed-gcc (xpack)

TOOLCHAIN ?= riscv-none-elf
TOOLCHAIN_PATH ?= /c/Users/AnlangZ/ZCodeProject/mouse-forwarder/tools/riscv-gcc/xpack-riscv-none-elf-gcc-14.2.0-2/bin

CC       = $(TOOLCHAIN)-gcc
OBJCOPY  = $(TOOLCHAIN)-objcopy
OBJDUMP  = $(TOOLCHAIN)-objdump
SIZE     = $(TOOLCHAIN)-size

# MCU settings
MCU      = ch32v305rbt6
FLASH    = 128K
RAM      = 32K

# Directories
LD_DIR   = .
OBJ_DIR  = obj

# Source files
SRCS = \
	main.c \
	ch32v30x_it.c \
	ch32v30x_usbfs_device.c \
	ch32v30x_usbhs_device.c \
	system_ch32v30x.c \
	usbd_composite_km.c \
	User/syscalls.c

# Object files
OBJS = $(SRCS:%.c=$(OBJ_DIR)/%.o)

# Include paths
INCLUDES = \
	-I. \
	-I./USB_Device \
	-I./UART \
	-I./User

# Linker script
LDSCRIPT = $(LD_DIR)/link.ld

# Compiler flags
CFLAGS = \
	-march=rv32imac \
	-mabi=ilp32 \
	-msmall-data-limit=8 \
	-mno-save-restore \
	-Os \
	-ffunction-sections \
	-fdata-sections \
	-fomit-frame-pointer \
	-Wall \
	$(INCLUDES) \
	-DCH32V305 \
	-DCH32V30x_D8C

# Linker flags
LDFLAGS = \
	-march=rv32imac \
	-mabi=ilp32 \
	-msmall-data-limit=8 \
	-mno-save-restore \
	-Os \
	-ffunction-sections \
	-fdata-sections \
	-Wl,--gc-sections \
	-Wl,-Map=$(OBJ_DIR)/firmware.map \
	-T$(LDSCRIPT)

# Targets
all: $(OBJ_DIR)/firmware.hex

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/firmware.elf: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ -lm
	$(SIZE) $@

$(OBJ_DIR)/firmware.hex: $(OBJ_DIR)/firmware.elf
	$(OBJCOPY) -O ihex $< $@
	$(OBJCOPY) -O binary $(OBJ_DIR)/firmware.elf $(OBJ_DIR)/firmware.bin
	@echo "=== Build complete ==="
	@$(SIZE) $<

flash: $(OBJ_DIR)/firmware.hex
	wchisptool -c CH32V305 -e -f $<

clean:
	rm -rf $(OBJ_DIR)

.PHONY: all clean flash