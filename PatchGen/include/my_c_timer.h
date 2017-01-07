/* Copyright (C) 2016 Andreas Doumanoglou 
 * You may use, distribute and modify this code under the terms
 * included in the LICENSE.txt
 *
 * Timer helper
 */
#ifndef MY_C_TIMER_H
#define MY_C_TIMER_H

#include <iostream>

class my_c_timer{

    struct timeval tv_;

 public:
    void reset(){
        gettimeofday(&tv_, NULL);
    }

    void print_time(const std::string &msg){
        struct timeval tv2;
        gettimeofday(&tv2, NULL);
        std::cout << msg << " -- Time elapsed: " << ((double)tv2.tv_usec - (double)tv_.tv_usec)/1000000.0 << std::endl;
    }

};

#endif
