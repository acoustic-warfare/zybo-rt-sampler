/******************************************************************************
 * Title                 :   Calculate antenna coefficients for delay
 * Filename              :   antenna/antenna.h
 * Author                :   Irreq
 * Origin Date           :   20/06/2023
 * Version               :   1.0.0
 * Compiler              :   gcc (GCC) 9.5.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 Functions for python to call when calculating coefficients

*/

/*
Python wrapper used to initate the antenna
*/
void py_antenna_wrapper(double azimuth,   // Horizontal
                        double elevation, // Vertical
                        int columns,
                        int rows,
                        float distance, // Distance between elements
                        float fs,
                        float propagation_speed,
                        float *coefficients);

/*
Calculate the different coefficients depeding on the antenna's direction and size
for a single antenna
*/
void directional_antenna_delay_coefficients(double azimuth,   // Horizontal
                                            double elevation, // Vertical
                                            int columns,
                                            int rows,
                                            float distance, // Distance between elements
                                            float fs,
                                            float propagation_speed,
                                            float **coefficients);