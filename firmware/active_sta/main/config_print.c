#include "config_print.h"
#include <stdio.h>
#include "sdkconfig.h"
#include "esp_log.h"
#include "config.h"

extern const char *TAG;

#define PROJECT_NAME "ACTIVE_STA"

void config_print() {
    ESP_LOGI(TAG, "=== Configuration ===");
    ESP_LOGI(TAG, "Project: %s", PROJECT_NAME);
    ESP_LOGI(TAG, "WiFi SSID: %s", WIFI_SSID);
    ESP_LOGI(TAG, "WiFi Channel: %d", WIFI_CHANNEL);
    ESP_LOGI(TAG, "Collect CSI: %s", SHOULD_COLLECT_CSI ? "Yes" : "No");
    ESP_LOGI(TAG, "Collect Only LLTF: %s", SHOULD_COLLECT_ONLY_LLTF ? "Yes" : "No");
    ESP_LOGI(TAG, "Send CSI to Serial: %s", SEND_CSI_TO_SERIAL ? "Yes" : "No");
    ESP_LOGI(TAG, "Send CSI to SD: %s", SEND_CSI_TO_SD ? "Yes" : "No");
    ESP_LOGI(TAG, "==================");
#ifdef WIFI_CHANNEL
    printf("WIFI_CHANNEL: %d\n", WIFI_CHANNEL);
#endif
#ifdef ESP_WIFI_SSID
    printf("ESP_WIFI_SSID: %s\n", ESP_WIFI_SSID);
#endif
#ifdef ESP_WIFI_PASS
    printf("ESP_WIFI_PASSWORD: %s\n", ESP_WIFI_PASS);
#endif
    printf("-----------------------\n");
}
