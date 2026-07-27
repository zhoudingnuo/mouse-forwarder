/**
 * @file    mouse_forwarder.c
 * @brief   鼠标转发器核心逻辑实现
 * 
 * 接收 USB1 (FS CDC) 传来的串口数据包,
 * 解析后通过 USB2 (HS HID) 发送鼠标报告。
 */

#include "debug.h"
#include "mouse_forwarder.h"
#include "protocol.h"

/* 当前鼠标报告 */
static MouseReport_t s_current_report = {0};

/* 统计信息 */
static volatile uint32_t s_packets_received = 0;
static volatile uint32_t s_packets_sent = 0;
static volatile uint32_t s_invalid_packets = 0;

/* 上次心跳时间 */
static volatile uint32_t s_last_heartbeat = 0;

/*********************************************************************
 * @fn      MouseForwarder_Init
 * @brief   初始化鼠标转发器
 */
void MouseForwarder_Init(void)
{
    s_current_report.buttons = 0;
    s_current_report.x = 0;
    s_current_report.y = 0;
    s_current_report.wheel = 0;
    s_packets_received = 0;
    s_packets_sent = 0;
    s_invalid_packets = 0;
    s_last_heartbeat = 0;
}

/*********************************************************************
 * @fn      MouseForwarder_ProcessPacket
 * @brief   处理 USB1 收到的数据包
 * @param   data  数据指针
 * @param   len   数据长度
 */
void MouseForwarder_ProcessPacket(uint8_t *data, uint16_t len)
{
    SerialPacket_t pkt;
    MouseReport_t report;

    /* 检查数据长度 */
    if (len < sizeof(SerialPacket_t))
    {
        s_invalid_packets++;
        printf("ERR: Packet too short (%d bytes)\r\n", len);
        return;
    }

    /* 可能收到多个包, 逐个处理 */
    uint16_t offset = 0;
    while (offset + sizeof(SerialPacket_t) <= len)
    {
        /* 查找帧头 */
        if (data[offset] != PKG_HEADER)
        {
            offset++;
            s_invalid_packets++;
            continue;
        }

        /* 拷贝数据包 */
        for (uint8_t i = 0; i < sizeof(SerialPacket_t); i++)
        {
            ((uint8_t *)&pkt)[i] = data[offset + i];
        }

        /* 验证数据包 */
        if (!validate_packet(&pkt))
        {
            offset++;
            s_invalid_packets++;
            continue;
        }

        /* 有效数据包 - 转换为 HID 报告 */
        packet_to_report(&pkt, &report);
        s_current_report.buttons = report.buttons;

        /* 通过 USB2 HS 发送 HID 鼠标报告 */
        USB2_SendMouseReport(&report);
        s_packets_received++;
        s_packets_sent++;

        offset += sizeof(SerialPacket_t);
    }

    /* 调试输出 (每 100 包打印一次) */
    if (s_packets_received % 100 == 0)
    {
        printf("Pkt:%lu Sent:%lu Invalid:%lu\r\n",
               (unsigned long)s_packets_received,
               (unsigned long)s_packets_sent,
               (unsigned long)s_invalid_packets);
    }
}

/*********************************************************************
 * @fn      MouseForwarder_Heartbeat
 * @brief   心跳维护 - 保持 HID 连接活跃
 */
void MouseForwarder_Heartbeat(void)
{
    /* 心跳预留 - 无需额外操作 */
}