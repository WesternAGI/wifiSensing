idf_component_register(
    SRCS 
        "main.cc"
        "config_print.c"
    INCLUDE_DIRS "."
    REQUIRES 
        freertos 
        nvs_flash 
        esp_wifi 
        esp_event 
        esp_netif 
        esp_http_client
)