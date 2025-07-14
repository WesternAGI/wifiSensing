#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_spi_flash.h"
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
 *
 * idf.py -p /dev/ttyUSB0 flash monitor | grep "CSI_DATA" >> ../mc29b_myself_working.csv
*/

#define LOG_LOCAL_LEVEL ESP_LOG_WARN

#define ESP_WIFI_SSID      "mywifi_ssid"
#define ESP_WIFI_PASS      "mywifi_pass"
#define MAX_STA_CONN       16


#define WIFI_CHANNEL 1



#define SHOULD_COLLECT_CSI 1



#define SHOULD_COLLECT_ONLY_LLTF 0



#define SEND_CSI_TO_SERIAL 1



#define SEND_CSI_TO_SD 0


/* FreeRTOS event group to signal when we are connected*/
static EventGroupHandle_t s_wifi_event_group;

static const char *TAG = "Active CSI collection (AP)";

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

void softap_init() {
    s_wifi_event_group = xEventGroupCreate();

    // Initialize TCP/IP and event loop
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    
    // Create default AP interface
    esp_netif_t *ap_netif = esp_netif_create_default_wifi_ap();
    assert(ap_netif);

    // Configure WiFi with default settings
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // Register event handler
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                      ESP_EVENT_ANY_ID,
                                                      &wifi_event_handler,
                                                      NULL,
                                                      NULL));

    // Configure AP settings
    wifi_ap_config_t wifi_ap_config = {};
    wifi_ap_config.channel = WIFI_CHANNEL;
    wifi_ap_config.authmode = WIFI_AUTH_WPA2_PSK;  // Use WPA2 only (more compatible than WPA/WPA2 mixed)
    wifi_ap_config.max_connection = 4;  // Reduced from 16 to 4 for better stability
    wifi_ap_config.beacon_interval = 100;  // 100ms beacon interval (default)
    wifi_ap_config.ssid_hidden = 0;  // Make SSID visible
    
    // Set country code (this affects available channels and power)
    wifi_country_t country = {
        .cc = "US",
        .schan = 1,
        .nchan = 11,
        .max_tx_power = 84,
        .policy = WIFI_COUNTRY_POLICY_AUTO
    };
    ESP_ERROR_CHECK(esp_wifi_set_country(&country));

    // Configure WiFi
    wifi_config_t wifi_config = {
        .ap = wifi_ap_config,
    };

    // Copy SSID and password
    strlcpy((char *)wifi_config.ap.ssid, ESP_WIFI_SSID, sizeof(ESP_WIFI_SSID));
    strlcpy((char *)wifi_config.ap.password, ESP_WIFI_PASS, sizeof(ESP_WIFI_PASS));

    // If no password is set, use open authentication
    if (strlen(ESP_WIFI_PASS) == 0) {
        wifi_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    // Set WiFi mode to AP
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    
    // Configure WiFi parameters
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    
    // Set WiFi bandwidth to 20MHz for better compatibility
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_AP, WIFI_BW_HT20));
    
    // Enable AMPDU RX for better performance
    ESP_ERROR_CHECK(esp_wifi_stop());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));  // Disable power save for better performance
    
    // Start WiFi
    ESP_ERROR_CHECK(esp_wifi_start());

    // Set maximum WiFi transmit power (20dBm)
    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(84));  // 84 = 20dBm

    ESP_LOGI(TAG, "softap_init finished. SSID:%s password:%s channel:%d", 
             ESP_WIFI_SSID, ESP_WIFI_PASS, WIFI_CHANNEL);
}

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

extern "C" void app_main() {

    printf("Active CSI collection (AP) started.\n");
    config_print();
    nvs_init();
    sd_init();
    softap_init();

#if !(SHOULD_COLLECT_CSI)
    printf("CSI will not be collected. Check `idf.py menuconfig  # > ESP32 CSI Tool Config` to enable CSI");
#else 
    printf("CSI will be collected.\n");
#endif

    csi_init((char *) "AP");
}