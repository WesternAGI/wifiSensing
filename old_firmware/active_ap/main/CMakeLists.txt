idf_component_register(
        SRCS main.cc
        INCLUDE_DIRS ""
        REQUIRES freertos esp_wifi spi_flash esp_http_server nvs_flash fatfs
)