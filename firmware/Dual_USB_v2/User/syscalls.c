/**
 * syscalls.c - Newlib 系统调用桩
 * 提供 _close, _fstat, _isatty, _lseek, _read
 * _write 和 _sbrk 在 debug.c 中已定义
 */
#include <sys/stat.h>
#include <errno.h>

int _close(int file) { return -1; }
int _fstat(int file, struct stat *st) { st->st_mode = S_IFCHR; return 0; }
int _isatty(int file) { return 1; }
int _lseek(int file, int ptr, int dir) { return 0; }
int _read(int file, char *ptr, int len) { return 0; }