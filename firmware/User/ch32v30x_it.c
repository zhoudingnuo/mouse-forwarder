/**
 * @file    ch32v30x_it.c
 * @brief   CH32V305 中断处理
 */

#include "debug.h"
#include "mouse_forwarder.h"

/*********************************************************************
 * @fn      NMI_Handler
 * @brief   不可屏蔽中断处理
 */
void NMI_Handler(void)
{
    printf("NMI_Handler\r\n");
}

/*********************************************************************
 * @fn      HardFault_Handler
 * @brief   硬件错误处理
 */
void HardFault_Handler(void)
{
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\r\n");
    printf("  Hard Fault!\r\n");
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\r\n");

    while (1) { }
}

/*********************************************************************
 * @fn      SysTick_Handler
 * @brief   系统滴答定时器中断
 */
void SysTick_Handler(void)
{
}