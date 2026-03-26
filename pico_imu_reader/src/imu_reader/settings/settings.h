#ifndef SETTINGS_H
#define SETTINGS_H

#include <stdbool.h>
#include "imu_reader.h"

#define SETTINGS_BUF_LEN (256)

void wait_for_settings(void);
bool settings_are_ready(void);

#endif //SETTINGS_H
