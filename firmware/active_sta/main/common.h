#ifndef COMMON_H
#define COMMON_H

#include <stdbool.h>  // For bool type in C

#ifdef __cplusplus
extern "C" {
#endif

// Common logging tag for the application
extern const char *TAG;

// Common function declarations
void nvs_init(void);
void sd_init(void);
void socket_transmitter_sta_loop(bool* is_wifi_connected);

// csi_init is already declared in csi_component.h
// so we don't redeclare it here to avoid conflicts

#ifdef __cplusplus
}
#endif

#endif // COMMON_H
