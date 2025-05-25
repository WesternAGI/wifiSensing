#include "common.h"
#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include <lwip/netdb.h>
#include "esp_log.h"

#define HOST_IP_ADDR CONFIG_HOST_IP_ADDR
#define PORT CONFIG_HOST_PORT

void socket_transmitter_sta_loop(bool* is_wifi_connected) {
    struct sockaddr_in dest_addr;
    int sock = 0;
    bool connected = false;
    
    while (1) {
        if (*is_wifi_connected && !connected) {
            // Create socket
            if ((sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP)) < 0) {
                ESP_LOGE(TAG, "Unable to create socket");
                vTaskDelay(1000 / portTICK_PERIOD_MS);
                continue;
            }
            
            // Set socket timeout
            struct timeval timeout;
            timeout.tv_sec = 5;
            timeout.tv_usec = 0;
            setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            
            // Configure the destination address
            dest_addr.sin_addr.s_addr = inet_addr(HOST_IP_ADDR);
            dest_addr.sin_family = AF_INET;
            dest_addr.sin_port = htons(PORT);
            
            // Connect to the server
            ESP_LOGI(TAG, "Socket connecting to %s:%d", HOST_IP_ADDR, PORT);
            int err = connect(sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
            if (err != 0) {
                ESP_LOGE(TAG, "Socket unable to connect: errno %d", errno);
                close(sock);
                vTaskDelay(1000 / portTICK_PERIOD_MS);
                continue;
            }
            
            ESP_LOGI(TAG, "Successfully connected to server");
            connected = true;
        } 
        else if (!*is_wifi_connected && connected) {
            // WiFi disconnected, close the socket
            if (sock != -1) {
                close(sock);
                sock = -1;
            }
            connected = false;
            ESP_LOGI(TAG, "Disconnected from server");
        }
        
        if (connected) {
            // Example: Send a ping message
            const char* message = "PING\n";
            int err = send(sock, message, strlen(message), 0);
            if (err < 0) {
                ESP_LOGE(TAG, "Error occurred during sending: errno %d", errno);
                close(sock);
                connected = false;
            }
        }
        
        vTaskDelay(5000 / portTICK_PERIOD_MS); // Wait 5 seconds between operations
    }
}
