/*
   Copyright 2013-2014 Бондаренко Иван Юрьевич

   Этот файл — часть NNSys.

   NNSys - свободная программа: вы можете перераспространять её и/или изменять
её на условиях Стандартной общественной лицензии GNU в том виде, в каком она
была опубликована Фондом свободного программного обеспечения; либо версии 3
лицензии, либо (по вашему выбору) любой более поздней версии.

   NNSys распространяется в надежде, что она будет полезной, но БЕЗО ВСЯКИХ
ГАРАНТИЙ; даже без неявной гарантии ТОВАРНОГО ВИДА или ПРИГОДНОСТИ ДЛЯ
ОПРЕДЕЛЕННЫХ ЦЕЛЕЙ. Подробнее см. в Стандартной общественной лицензии GNU.

   Вы должны были получить копию Стандартной общественной лицензии GNU вместе с
этой программой. Если это не так, см. http://www.gnu.org/licenses/gpl.html.
*/

#define _USE_MATH_DEFINES
#include <cfloat>
#include <climits>
#include <cmath>

#include "randlib.h"

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0 / IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1 + IMM1 / NTAB)
#define EPS 1.2e-7
#define RNMX (1.0 - EPS)

long g_seedForRandom = -1;

double generate_random_value()
{
    int j;
    long k;
    static long idum2 = 123456789;
    static long iy=0;
    static long iv[NTAB];
    double temp, result;

    if (g_seedForRandom <= 0)
    {
        if ((-g_seedForRandom) < 1)
        {
            g_seedForRandom = 1;
        }
        else
        {
            g_seedForRandom = -g_seedForRandom;
        }
        idum2 = g_seedForRandom;
        for (j = NTAB + 7; j >= 0; j--)
        {
            k= g_seedForRandom / IQ1;
            g_seedForRandom = IA1 * (g_seedForRandom - k * IQ1) - k * IR1;
            if (g_seedForRandom < 0)
            {
                g_seedForRandom += IM1;
            }
            if (j < NTAB)
            {
                iv[j] = g_seedForRandom;
            }
        }
        iy=iv[0];
    }
    k = g_seedForRandom / IQ1;
    g_seedForRandom = IA1 * (g_seedForRandom - k * IQ1) - k * IR1;
    if (g_seedForRandom < 0)
    {
        g_seedForRandom += IM1;
    }
    k = idum2 / IQ2;
    idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
    if (idum2 < 0)
    {
        idum2 += IM2;
    }
    j = iy / NDIV;
    iy = iv[j] - idum2;
    iv[j] = g_seedForRandom;
    if (iy < 1)
    {
        iy += IMM1;
    }
    if ((temp = AM * iy) > RNMX)
    {
        result = RNMX;
    }
    else
    {
        result = temp;
    }

    return result;
}

double generate_random_value(double min_val, double max_val)
{
    if (fabs(max_val - min_val) <= FLT_EPSILON)
    {
        return min_val;
    }
    double res = generate_random_value();
    return (min_val + res * (max_val - min_val));
}

double generate_normal_random_value()
{
    static int iset = 0;
    static double gset;
    double fac, rsq, v1, v2, result;

    if (g_seedForRandom < 0)
    {
        iset = 0;
    }
    if (iset == 0)
    {
        do {
            v1 = 2.0 * generate_random_value() - 1.0;
            v2 = 2.0 * generate_random_value() - 1.0;
            rsq = v1 * v1 + v2 * v2;
        } while ((rsq >= 1.0) || (fabs(rsq) <= FLT_EPSILON));
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        iset = 1;
        result = v2 * fac;
    }
    else
    {
        iset = 0;
        result = gset;
    }

    return result;
}

void initialize_random_generator(long value)
{
    if (value < 0)
    {
        g_seedForRandom = value;
    }
    else
    {
        g_seedForRandom = -value;
        if (g_seedForRandom >= 0)
        {
            g_seedForRandom = -1;
        }
    }
}
