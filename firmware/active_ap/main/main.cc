/*
 * File: main.cc
 * Project: ESP32 Active Access Point (AP) CSI Collection
 * Description:
 *   This file contains the main logic for the ESP32 operating in Access Point mode (AP) to collect Channel State Information (CSI).
 *   It sets up the ESP32 as a WiFi AP, manages connection and disconnection events, collects CSI data, and coordinates with other system components.
 *   The code uses FreeRTOS for task management and the ESP-IDF framework for hardware abstraction.
 *
 * Key Responsibilities:
 *   - Initialize WiFi in AP mode and manage connection events for stations
 *   - Collect CSI data as configured
 *   - Initialize and coordinate with system components (NVS, SD, CSI, Time, Input, Sockets)
 *   - Print configuration and build info for diagnostics
 *
 * Usage:
 *   - Configure WiFi and CSI settings via menuconfig or by editing the defines below
 *   - Build and flash to ESP32 device
 *
 * Author: [Your Name or Team]
 * Date: [Date]
 */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_system.h"
#include "esp_spi_flash.h"
#include "esp_flash.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "lwip/err.h"
#include "lwip/sys.h"

#include "../../_components/nvs_component.h"
#include "../../_components/sd_component.h"
#include "../../_components/csi_component.h"
#include "../../_components/time_component.h"
#include "../../_components/input_component.h"
#include "../../_components/sockets_component.h"

/*
 * The examples use WiFi configuration that you can set via 'idf.py menuconfig'.
 *
 * If you'd rather not, just change the below entries to strings with
 * the config you want - ie #define ESP_WIFI_SSID "mywifissid"
 */
// WiFi credentials and maximum allowed station connections (set via menuconfig or hardcoded here)
#define ESP_WIFI_SSID      CONFIG_ESP_WIFI_SSID
#define ESP_WIFI_PASS      CONFIG_ESP_WIFI_PASSWORD
#define MAX_STA_CONN       16

#ifdef CONFIG_WIFI_CHANNEL
#define WIFI_CHANNEL CONFIG_WIFI_CHANNEL
#else
#define WIFI_CHANNEL 6
#endif

#ifdef CONFIG_SHOULD_COLLECT_CSI
#define SHOULD_COLLECT_CSI 1
#else
#define SHOULD_COLLECT_CSI 0
#endif

#ifdef CONFIG_SHOULD_COLLECT_ONLY_LLTF
#define SHOULD_COLLECT_ONLY_LLTF 1
#else
#define SHOULD_COLLECT_ONLY_LLTF 0
#endif

#ifdef CONFIG_SEND_CSI_TO_SERIAL
#define SEND_CSI_TO_SERIAL 1
#else
#define SEND_CSI_TO_SERIAL 0
#endif

#ifdef CONFIG_SEND_CSI_TO_SD
#define SEND_CSI_TO_SD 1
#else
#define SEND_CSI_TO_SD 0
#endif

/* FreeRTOS event group to signal when we are connected*/
// FreeRTOS event group handle to signal WiFi connection events
static EventGroupHandle_t s_wifi_event_group;

// Logging tag for ESP-IDF logs
static const char *TAG = "Active CSI collection (AP)";

// Define the mutex for CSI component
SemaphoreHandle_t mutex = NULL;

// WiFi event handler for AP mode: logs station connect/disconnect events
static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t* event = (wifi_event_ap_staconnected_t*) event_data;
        ESP_LOGI(TAG, "station " MACSTR " join, AID=%d",
                 MAC2STR(event->mac), event->aid);
    } else if (event_id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t* event = (wifi_event_ap_stadisconnected_t*) event_data;
        ESP_LOGI(TAG, "station " MACSTR " leave, AID=%d",
                 MAC2STR(event->mac), event->aid);
    }
}

// Initialize WiFi in Access Point (AP) mode and register event handlers
void softap_init() {
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        NULL));

    wifi_ap_config_t wifi_ap_config = {};
    wifi_ap_config.channel = WIFI_CHANNEL;
    wifi_ap_config.authmode = WIFI_AUTH_WPA_WPA2_PSK;
    wifi_ap_config.max_connection = MAX_STA_CONN;

    wifi_config_t wifi_config = {
            .ap = wifi_ap_config,
    };

    strlcpy((char *) wifi_config.ap.ssid, ESP_WIFI_SSID, sizeof(ESP_WIFI_SSID));
    strlcpy((char *) wifi_config.ap.password, ESP_WIFI_PASS, sizeof(ESP_WIFI_PASS));

    if (strlen(ESP_WIFI_PASS) == 0) {
        wifi_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    esp_wifi_set_ps(WIFI_PS_NONE);

    ESP_LOGI(TAG, "softap_init finished. SSID:%s password:%s", ESP_WIFI_SSID, ESP_WIFI_PASS);
}

// Print configuration and build info to the console
void config_print() {
    printf("\n\n\n\n\n\n\n\n");
    printf("-----------------------\n");
    printf("ESP32 CSI Tool Settings\n");
    printf("-----------------------\n");
    printf("PROJECT_NAME: %s\n", "ACTIVE_AP");
    printf("CONFIG_ESPTOOLPY_MONITOR_BAUD: %d\n", CONFIG_ESPTOOLPY_MONITOR_BAUD);
    printf("CONFIG_ESP_CONSOLE_UART_BAUDRATE: %d\n", CONFIG_ESP_CONSOLE_UART_BAUDRATE);
    printf("IDF_VER: %s\n", IDF_VER);
    printf("-----------------------\n");
    printf("WIFI_CHANNEL: %d\n", WIFI_CHANNEL);
    printf("ESP_WIFI_SSID: %s\n", ESP_WIFI_SSID);
    printf("ESP_WIFI_PASSWORD: %s\n", ESP_WIFI_PASS);
    printf("SHOULD_COLLECT_CSI: %d\n", SHOULD_COLLECT_CSI);
    printf("SHOULD_COLLECT_ONLY_LLTF: %d\n", SHOULD_COLLECT_ONLY_LLTF);
    printf("SEND_CSI_TO_SERIAL: %d\n", SEND_CSI_TO_SERIAL);
    printf("SEND_CSI_TO_SD: %d\n", SEND_CSI_TO_SD);
    printf("-----------------------\n");
    printf("\n\n\n\n\n\n\n\n");
}

/*
 * Main entry point for the ESP32 application (called by ESP-IDF).
 * Initializes components, WiFi AP mode, CSI collection, and prints configuration info.
 */
extern "C" void app_main() {
    // Create the mutex for CSI component
    mutex = xSemaphoreCreateMutex();
    if (mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create mutex");
        return;
    }
    
    config_print();
    // esp_wifi_set_protocol(ifx,WIFI_PROTOCOL_11N);
    nvs_init();
    sd_init();
    softap_init();

#if !(SHOULD_COLLECT_CSI)
    printf("CSI will not be collected. Check `idf.py menuconfig  # > ESP32 CSI Tool Config` to enable CSI");
#endif

    csi_init((char *) "AP");
}
