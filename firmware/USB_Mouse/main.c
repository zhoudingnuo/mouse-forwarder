/**
 * @file    main.c
 * @brief   CH32V305 鼠标转发器 - 主程序
 * 
 * 基于 WCH 官方 CompositeKM 示例
 * 通过 USART2 (PA3) 接收 PC 发来的鼠标数据包
 * 通过 USBFS 转发为 HID 鼠标报告
 * 
 * 板载 LED 闪烁指示运行状态
 * 
 * 通信协议 (6 字节):
 *   [0xAA] [flags] [dx] [dy] [wheel] [checksum]
 */

#include "debug.h"
#include "ch32v30x_usbfs_device.h"
#include "usbd_composite_km.h"

extern volatile uint8_t USBFS_DevEnumStatus;

/* LED 闪烁 - 测试多个可能的引脚 */
static void LED_Blink_All(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;
    
    /* 使能所有 GPIO 时钟 */
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA | RCC_APB2Periph_GPIOB | 
                           RCC_APB2Periph_GPIOC | RCC_APB2Periph_GPIOD, ENABLE);
    
    /* 配置 PC13 (CH32V305 板载 LED 常见引脚) */
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_13;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOC, &GPIO_InitStructure);
    
    /* 配置 PB2 (备用 LED 引脚) */
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;
    GPIO_Init(GPIOB, &GPIO_InitStructure);
    
    /* 配置 PA15 (备用 LED 引脚) */
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_15;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
    
    /* 闪烁 5 次, 每次 100ms */
    for(int i = 0; i < 10; i++)
    {
        GPIO_SetBits(GPIOC, GPIO_Pin_13);
        GPIO_SetBits(GPIOB, GPIO_Pin_2);
        GPIO_SetBits(GPIOA, GPIO_Pin_15);
        Delay_Ms(100);
        GPIO_ResetBits(GPIOC, GPIO_Pin_13);
        GPIO_ResetBits(GPIOB, GPIO_Pin_2);
        GPIO_ResetBits(GPIOA, GPIO_Pin_15);
        Delay_Ms(100);
    }
}

int main(void)
{
    /* 系统初始化 */
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
    SystemCoreClockUpdate();
    Delay_Init();
    USART_Printf_Init(115200);

    printf("========================================\r\n");
    printf("  CH32V305 Mouse Forwarder v1.0\r\n");
    printf("========================================\r\n");
    printf("SystemClk: %d\r\n", SystemCoreClock);
    printf("ChipID: %08x\r\n", DBGMCU_GetCHIPID());

    /* LED 闪烁测试 - 启动时闪 5 次 */
    printf("LED blink test...\r\n");
    LED_Blink_All();
    printf("LED test done\r\n");

    /* 初始化 USART2 接收鼠标数据 */
    USART2_Init(115200);
    printf("USART2 Init OK (115200 baud)\r\n");

    /* 初始化 USBFS HID 鼠标 */
    USBFS_RCC_Init();
    USBFS_Device_Init(ENABLE);
    printf("USBFS HID Mouse Init OK\r\n");

    printf("System Ready!\r\n");

    uint32_t blink_cnt = 0;
    while(1)
    {
        if(USBFS_DevEnumStatus)
        {
            /* 处理从 USART2 接收的鼠标数据 */
            Mouse_Data_Handle();
            
            /* LED 心跳 - 每 50000 次循环翻转一次 */
            if(++blink_cnt >= 50000)
            {
                blink_cnt = 0;
                GPIOC->OUTDR ^= GPIO_Pin_13;  /* 翻转 PC13 */
            }
        }
        else
        {
            /* USB 未连接时快速闪烁 */
            Delay_Ms(200);
            GPIOC->OUTDR ^= GPIO_Pin_13;
        }
    }
}