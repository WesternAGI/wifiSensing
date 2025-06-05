#include "common.h"
#include "driver/sdmmc_host.h"
#include "driver/sdspi_host.h"
#include "sdmmc_cmd.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"

static const char *SD_TAG = "SD_UTILS";

void sd_init() {
    ESP_LOGI(SD_TAG, "Initializing SD card");
    
    // Initialize the SPI bus
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    
    // Configuration for the SD card interface
    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = CONFIG_EXAMPLE_PIN_CS;
    slot_config.host_id = host.slot;
    
    // Initialize the SPI bus with default configuration
    // The actual pins are configured through the slot_config above
    ESP_ERROR_CHECK(spi_bus_initialize(host.slot, NULL, 1));
    
    // Options for mounting the filesystem
    esp_vfs_fat_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024
    };
    
    sdmmc_card_t *card;
    esp_err_t ret = esp_vfs_fat_sdspi_mount("/sdcard", &host, &slot_config, &mount_config, &card);
    
    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(SD_TAG, "Failed to mount filesystem. "
                     "If you want the card to be formatted, set format_if_mount_failed = true.");
        } else {
            ESP_LOGE(SD_TAG, "Failed to initialize the card (0x%x)", ret);
        }
        return;
    }
    
    // Card has been initialized, print its properties
    sdmmc_card_print_info(stdout, card);
    ESP_LOGI(SD_TAG, "SD card initialized successfully");
}
