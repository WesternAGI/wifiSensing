/*
 * File: main.cc
 * Project: ESP32 Active Station (STA) CSI Collection
 * Description:
 *   This file contains the main logic for the ESP32 operating in Station mode (STA) to collect Channel State Information (CSI).
 *   It handles WiFi connection, event management, CSI data collection, and communication with other system components.
 *   The code uses FreeRTOS for task management and the ESP-IDF framework for hardware abstraction.
 *
 * Key Responsibilities:
 *   - Initialize WiFi in STA mode and manage connection events
 *   - Collect CSI data as configured
 *   - Transmit CSI or other data over sockets or store to SD as per configuration
 *   - Initialize and coordinate with system components (NVS, SD, CSI, Time, Input, Sockets)
 *
 * Usage:
 *   - Configure WiFi and CSI settings via menuconfig or by editing the defines below
 *   - Build and flash to ESP32 device
 *
 * Author: [Your Name or Team]
 * Date: [Date]
 */

#include <stdio.h>
#include <cstring>
#include "esp_wifi.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_http_client.h"
#include "../_components/csi_component.h"

SemaphoreHandle_t mutex = xSemaphoreCreateMutex();

#include "config.h"

// Function declarations
bool is_wifi_connected();

// Task handle for the socket transmitter task
TaskHandle_t xHandle = NULL;

// Logging tag for ESP-IDF logs
static const char *TAG = "Active CSI collection (Station)";

// Include config_print header
#include "config_print.h"

void wifi_sta_init() {
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    wifi_config_t wifi_config = {};
    snprintf((char*)wifi_config.sta.ssid, sizeof(wifi_config.sta.ssid), "%s", WIFI_SSID);
    snprintf((char*)wifi_config.sta.password, sizeof(wifi_config.sta.password), "%s", WIFI_PASS);
    wifi_config.sta.channel = WIFI_CHANNEL;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "WiFi STA init done. SSID:%s", WIFI_SSID);
}

// Duplicate app_main removed. Only one definition exists below.

extern void time_set(char* data);
extern size_t strlcpy(char* dst, const char* src, size_t siz);
extern void socket_transmitter_sta_loop(bool* is_wifi_connected); 
extern void nvs_init();
extern void sd_init();
extern void csi_init(char* mode);

extern void vTask_socket_transmitter_sta_loop(void *pvParameters)
{
    bool wifi_connected = is_wifi_connected();
    socket_transmitter_sta_loop(&wifi_connected);
}

/*
 * The examples use WiFi configuration that you can set via 'idf.py menuconfig'.
 *
 * If you'd rather not, just change the below entries to strings with
 * the config you want - ie #define ESP_WIFI_SSID "mywifissid"
 */
// WiFi credentials (can be set via menuconfig or hardcoded here)
#define ESP_WIFI_SSID      CONFIG_ESP_WIFI_SSID
#define ESP_WIFI_PASS      CONFIG_ESP_WIFI_PASSWORD



// Configuration is now in config.h

/* FreeRTOS event group to signal when we are connected*/
// FreeRTOS event group handle to signal WiFi connection events
static EventGroupHandle_t s_wifi_event_group;

/* The event group allows multiple bits for each event, but we only care about one event
 * - are we connected to the AP with an IP? */
// Bit flag for WiFi connected event
const int WIFI_CONNECTED_BIT = BIT0;

// Logging tag for ESP-IDF logs

// HTTP event handler for time synchronization or other HTTP events
esp_err_t _http_event_handle(esp_http_client_event_t *evt) {
    switch (evt->event_id) {
        case HTTP_EVENT_ON_DATA:
            ESP_LOGI(TAG, "HTTP_EVENT_ON_DATA, len=%d", evt->data_len);
            if (!esp_http_client_is_chunked_response(evt->client)) {
                if (!real_time_set) {
                    char *data = (char *) malloc(evt->data_len + 1);
                    strncpy(data, (char *) evt->data, evt->data_len);
                    data[evt->data_len + 1] = '\0';
                    time_set(data);
                    free(data);
                }
            }
            break;
        default:
            break;
    }
    return ESP_OK;
}

//// en_sys_seq: see https://github.com/espressif/esp-idf/blob/master/docs/api-guides/wifi.rst#wi-fi-80211-packet-send for details
// Declaration for raw 802.11 packet transmission (if needed for advanced CSI use)
esp_err_t esp_wifi_80211_tx(wifi_interface_t ifx, const void *buffer, int len, bool en_sys_seq);

// WiFi and IP event handler for connection management
static void event_handler(void* arg, esp_event_base_t event_base,
                          int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "Retry connecting to the AP");
        esp_wifi_connect();
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got ip:" IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

// Helper function to check if WiFi is connected
bool is_wifi_connected() {
    return (xEventGroupGetBits(s_wifi_event_group) & WIFI_CONNECTED_BIT);
}



// Initialize WiFi in Station (STA) mode and register event handlers
void station_init() {
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());

    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_got_ip));

    wifi_sta_config_t wifi_sta_config = {};
    wifi_sta_config.channel = WIFI_CHANNEL;
    wifi_config_t wifi_config = {
            .sta = wifi_sta_config,
    };

    strlcpy((char *) wifi_config.sta.ssid, ESP_WIFI_SSID, sizeof(ESP_WIFI_SSID));
    strlcpy((char *) wifi_config.sta.password, ESP_WIFI_PASS, sizeof(ESP_WIFI_PASS));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    esp_wifi_set_ps(WIFI_PS_NONE);

    ESP_LOGI(TAG, "connect to ap SSID:%s password:%s", ESP_WIFI_SSID, ESP_WIFI_PASS);
}

#ifdef CONFIG_SHOULD_COLLECT_CSI
#define SHOULD_COLLECT_CSI 1
#else
#define SHOULD_COLLECT_CSI 0
#endif

/*
 * Main entry point for the ESP32 application (called by ESP-IDF).
 * Initializes components, WiFi, CSI collection, and starts the main data transmission task.
 */
#include "config_print.h"
#include "freertos/task.h"

extern "C" void app_main() {
    config_print();
    TaskHandle_t xHandle = NULL;
    // esp_wifi_set_protocol(ifx,WIFI_PROTOCOL_11N);
    nvs_init();
    sd_init();
    station_init();
    csi_init((char *) "STA");

#if !(SHOULD_COLLECT_CSI)
    printf("CSI will not be collected. Check `idf.py menuconfig  # > ESP32 CSI Tool Config` to enable CSI");
#endif

    xTaskCreatePinnedToCore(&vTask_socket_transmitter_sta_loop, "socket_transmitter_sta_loop",
                            10000, (void *) &is_wifi_connected, 100, &xHandle, 1); // xHandle now declared above
}
