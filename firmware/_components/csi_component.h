#ifndef ESP32_CSI_CSI_COMPONENT_H
#define ESP32_CSI_CSI_COMPONENT_H

#include "time_component.h"
#include "math.h"
#include <sstream>
#include <iostream>
#include "freertos/queue.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/uart.h"

char *project_type;

#define CSI_RAW 1
#define CSI_AMPLITUDE 0
#define CSI_PHASE 0

#define CSI_TYPE CSI_RAW

SemaphoreHandle_t mutex = xSemaphoreCreateMutex();

// Queue to transfer CSI packets from the Wi-Fi callback to a dedicated task
static QueueHandle_t csi_queue = nullptr;
static bool uart_initialized = false; // ensure UART is configured once

// Fast Wi-Fi CSI callback – only copies data and enqueues it
void _wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
    // Allocate a contiguous buffer large enough for the header struct and CSI bytes
    size_t total_size = sizeof(wifi_csi_info_t) + data->len;
    wifi_csi_info_t *copy = (wifi_csi_info_t *) malloc(total_size);
    if (!copy) {
        return; // allocation failed – drop packet
    }

    memcpy(copy, data, sizeof(wifi_csi_info_t));
    // Copy raw CSI bytes just after the struct
    copy->buf = (int8_t *) ((uint8_t *) copy + sizeof(wifi_csi_info_t));
    memcpy(copy->buf, data->buf, data->len);

    // Enqueue pointer to processing task (from non-ISR context this is safe)
    BaseType_t res = xQueueSend(csi_queue, &copy, 0);
    if (res != pdTRUE) {
        free(copy); // queue full – drop
    }
}


// Task running on a dedicated core to process and print CSI data
static void _init_uart_if_needed() {
#if SEND_CSI_TO_SERIAL
    if (!uart_initialized) {
        // UART0 already configured by ESP-IDF console, nothing to do.
        uart_initialized = true;
    }
#endif
}

void csi_processing_task(void *pvParameters) {
    wifi_csi_info_t *pkt;
    for (;;) {
        if (xQueueReceive(csi_queue, &pkt, portMAX_DELAY) == pdTRUE) {
            xSemaphoreTake(mutex, portMAX_DELAY);
            std::stringstream ss;

            wifi_csi_info_t d = *pkt;
            char mac[20] = {0};
            sprintf(mac, "%02X:%02X:%02X:%02X:%02X:%02X", d.mac[0], d.mac[1], d.mac[2], d.mac[3], d.mac[4], d.mac[5]);

            ss << "CSI_DATA," << project_type << "," << mac << ","
               << d.rx_ctrl.rssi << "," << d.rx_ctrl.rate << "," << d.rx_ctrl.sig_mode << "," << d.rx_ctrl.mcs << ","
               << d.rx_ctrl.cwb << "," << d.rx_ctrl.smoothing << "," << d.rx_ctrl.not_sounding << "," << d.rx_ctrl.aggregation << ","
               << d.rx_ctrl.stbc << "," << d.rx_ctrl.fec_coding << "," << d.rx_ctrl.sgi << "," << d.rx_ctrl.noise_floor << ","
               << d.rx_ctrl.ampdu_cnt << "," << d.rx_ctrl.channel << "," << d.rx_ctrl.secondary_channel << ","
               << d.rx_ctrl.timestamp << "," << d.rx_ctrl.ant << "," << d.rx_ctrl.sig_len << "," << d.rx_ctrl.rx_state << ","
               << real_time_set << "," << get_steady_clock_timestamp() << "," << d.len << ",[";

            int data_len = d.len;
            int8_t *my_ptr = d.buf;
#if CSI_RAW
            for (int i = 0; i < data_len; i++) {
                ss << (int) my_ptr[i] << " ";
            }
#endif
#if CSI_AMPLITUDE
            for (int i = 0; i < data_len / 2; i++) {
                ss << (int) sqrt(pow(my_ptr[i * 2], 2) + pow(my_ptr[(i * 2) + 1], 2)) << " ";
            }
#endif
#if CSI_PHASE
            for (int i = 0; i < data_len / 2; i++) {
                ss << (int) atan2(my_ptr[i * 2], my_ptr[(i * 2) + 1]) << " ";
            }
#endif
            ss << "]\n";

#if SEND_CSI_TO_SERIAL
            _init_uart_if_needed();
            std::string record = ss.str();
            const char *buf_ptr = record.c_str();
            size_t remaining = record.length();
            while (remaining) {
                int written = uart_write_bytes(UART_NUM_0, buf_ptr, remaining);
                if (written < 0) {
                    // UART error – drop the rest of this record
                    break;
                }
                remaining -= written;
                buf_ptr   += written;
                if (remaining) {
                    // Yield briefly to allow Wi-Fi tasks to run
                    vTaskDelay(1);
                }
            }
#endif
            xSemaphoreGive(mutex);
            free(pkt);
        }
    }
}

void _print_csi_csv_header() {
    char *header_str = (char *) "type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,real_time_set,real_timestamp,len,CSI_DATA\n";
    outprintf(header_str);
}

void csi_init(char *type) {

    // Check if the type is not "AP"
    if (strcmp(type, "AP") != 0) {
        ESP_LOGI("CSI_INIT", "Type is not AP, exiting csi_init.");
        return;
    }
    
    project_type = type;

    // Create queue & processing task the first time we initialise CSI
    if (csi_queue == nullptr) {
        _init_uart_if_needed();
        csi_queue = xQueueCreate(20, sizeof(void *));
        xTaskCreatePinnedToCore(csi_processing_task, "csi_proc", 8192, NULL, 5, NULL, 1);
    }

    ESP_ERROR_CHECK(esp_wifi_set_csi(1));

    // @See: https://github.com/espressif/esp-idf/blob/master/components/esp_wifi/include/esp_wifi_types.h#L401
    wifi_csi_config_t configuration_csi;
    configuration_csi.lltf_en = 1;
    configuration_csi.htltf_en = 1;
    configuration_csi.stbc_htltf2_en = 1;
    configuration_csi.ltf_merge_en = 1;
    configuration_csi.channel_filter_en = 0;
    configuration_csi.manu_scale = 0;

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&configuration_csi));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&_wifi_csi_cb, NULL));

    _print_csi_csv_header();

}

#endif //ESP32_CSI_CSI_COMPONENT_H