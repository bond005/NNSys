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

#include <cfloat>
#include <ctime>
#include <omp.h>
//#include <iostream> // for debug

#include <QFile>
#include <QDataStream>
#include <QVector>

#include "annlib.h"

#ifdef Q_WS_WIN
#define ISNAN(x) _isnan(x)
#define ISFINITE(x) _finite(x)
#else
#define ISNAN(x) isnan(x)
#define ISFINITE(x) finite(x)
#endif

using namespace std;

const float GOLD = 1.618034;
const float CGOLD = 0.3819660;

const int N_EXP = 20;
const float X_EXP[] = {
    -15.000000000000000,
     -8.497847557067871,
     -6.363110065460205,
     -5.093795299530029,
     -4.190978527069092,
     -3.491690874099731,
     -2.914491415023804,
     -2.426292657852173,
     -2.004758119583130,
     -1.630549907684326,
     -1.292358517646790,
     -0.985944092273712,
     -0.700374484062195,
     -0.436138868331909,
     -0.190974876284599,
      0.038653030991554,
      0.254159927368164,
      0.457662194967270,
      0.648983538150787,
      0.830329835414886,
      1.000000000000000
};
const float A_EXP[] = {
    0.000031312843930,
    0.000712073408067,
    0.003474863944575,
    0.009965231642127,
    0.021904963999987,
    0.041202854365110,
    0.069914579391479,
    0.109905727207661,
    0.163355544209480,
    0.233005508780479,
    0.321344256401062,
    0.431812524795532,
    0.568161785602570,
    0.732674419879913,
    0.928704261779785,
    1.159908533096313,
    1.429944872856140,
    1.741675257682800,
    2.098088264465332,
    2.500183105468750
};
const float B_EXP[] = {
    0.000469998572953,
    0.006254998035729,
    0.023834938183427,
    0.056895542889833,
    0.106934703886509,
    0.174316972494125,
    0.257997035980225,
    0.355027288198471,
    0.462181240320206,
    0.575748980045319,
    0.689914345741272,
    0.798829853534698,
    0.894325375556946,
    0.966075778007507,
    1.003512501716614,
    0.994575798511505,
    0.925943374633789,
    0.783276140689850,
    0.551970005035400,
    0.218098640441895
};
const float Y_EXP_0 = A_EXP[0] * X_EXP[0] + B_EXP[0];

#define SHFT(a, b, c, d) (a) = (b); (b) = (c); (c) = (d);
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

/*****************************************************************************/
/* Вычислить активационную функцию y(x) = 2x / (1 + abs(x)). */
/*****************************************************************************/
inline float activation(float x, TActivationKind kind)
{
    return ((kind == SIG) ? (2.0 * x / (1 + fabs(x))):x);
}

/*****************************************************************************/
/* Вычислить производную активационной функции y(x) по формуле:
   dy(x)         2.0
   ----- = ---------------.
    dx     (1 + abs(x))^2
*/
/*****************************************************************************/
inline float activation_derivative(float x, TActivationKind kind)
{
    float temp = 1.0 + fabs(x);
    return ((kind == SIG) ? (2.0 / (temp * temp)):1.0);
}

/*****************************************************************************/
/* Вычисление функции sign(x) */
/*****************************************************************************/
inline float sign(float x)
{
    if (x != 0.0)
    {
        if (x > 0.0)
        {
            x = 1.0;
        }
        else
        {
            x = -1.0;
        }
    }
    return x;
}

/* Вычислить приближённое значение функции y = exp(x) с помощтью таблицы
кусочно-линейной аппроксимации, заданной массивами A[0:9], B[0:9] - списками
параметров A и B в уравнениях прямых для всех 10 отрезков, на которые разбита
область определения функции. X-координаты границ этих отрезков заданы массивом
X[0:10]. */
inline float calc_exp(float x)
{
    register int i;
    if (x <= X_EXP[10]) // first = 0, last = 20, mid = 10
    {
        if (x <= X_EXP[5]) // first = 0, last = 10, mid = 5
        {
            if (x <= X_EXP[2]) // first = 0, last = 5, mid = 2
            {
                if (x <= X_EXP[1]) // first = 0, last = 2, mid = 1
                {
                    if (x <= X_EXP[0]) // first = 0, last = 1, mid = 0
                    {
                        i = -1;
                    }
                    else
                    {
                        i = 0;
                    }
                }
                else // first = 2, last = 2, mid = 2
                {
                    i = 1;
                }
            }
            else
            {
                if (x <= X_EXP[4]) // first = 3, last = 5, mid = 4
                {
                    if (x <= X_EXP[3]) // first = 3, last = 4, mid = 3
                    {
                        i = 2;
                    }
                    else // first = 4, last = 4, mid = 4
                    {
                        i = 3;
                    }
                }
                else // first = 5, last = 5, mid = 5
                {
                    i = 4;
                }
            }
        }
        else
        {
            if (x <= X_EXP[8]) // first = 6, last = 10, mid = 8
            {
                if (x <= X_EXP[7]) // first = 6, last = 8, mid = 7
                {
                    if (x <= X_EXP[6]) // first = 6, last = 7, mid = 6
                    {
                        i = 5;
                    }
                    else // first = 7, last = 7, mid = 7
                    {
                        i = 6;
                    }
                }
                else // first = 8, last = 8, mid = 8
                {
                    i = 7;
                }
            }
            else
            {
                if (x <= X_EXP[9]) // first = 9, last = 10, mid = 9
                {
                    i = 8;
                }
                else // first = 10, last = 10, mid = 10
                {
                    i = 9;
                }
            }
        }
    }
    else
    {
        if (x <= X_EXP[15]) // first = 11, last = 20, mid = 15
        {
            if (x <= X_EXP[13]) // first = 11, last = 15, mid = 13
            {
                if (x <= X_EXP[12]) // first = 11, last = 13, mid = 12
                {
                    if (x <= X_EXP[11]) // first = 11, last = 12, mid = 11
                    {
                        i = 10;
                    }
                    else // first = 12, last = 12, mid = 12
                    {
                        i = 11;
                    }
                }
                else // first = 13, last = 13, mid = 13
                {
                    i = 12;
                }
            }
            else
            {
                if (x <= X_EXP[14]) // first = 14, last = 15, mid = 14
                {
                    i = 13;
                }
                else // first = 15, last = 15, mid = 15
                {
                    i = 14;
                }
            }
        }
        else
        {
            if (x <= X_EXP[18]) // first = 16, last = 20, mid = 18
            {
                if (x <= X_EXP[17]) // first = 16, last = 18, mid = 17
                {
                    if (x <= X_EXP[16]) // first = 16, last = 17, mid = 16
                    {
                        i = 15;
                    }
                    else // first = 17, last = 17, mid = 17
                    {
                        i = 16;
                    }
                }
                else // first = 18, last = 18, mid = 18
                {
                    i = 17;
                }
            }
            else
            {
                if (x <= X_EXP[19]) // first = 19, last = 20, mid = 19
                {
                    i = 18;
                }
                else // first = 20, last = 20, mid = 20
                {
                    i = 19;
                }
            }
        }
    }
    return (i >= 0) ? (A_EXP[i] * x + B_EXP[i]) : Y_EXP_0;
}

/*****************************************************************************/
/*           РЕАЛИЗАЦИЯ КЛАССОВ, СВЯЗАННЫХ С ГЕНЕРАЦИЕЙ ИСКЛЮЧЕНИЙ           */
/*****************************************************************************/

const char* EANNError::what() const throw()
{
    return m_sErrorMessage.toStdString().c_str();
}

EMLPStructError::EMLPStructError() throw()
{
    m_sErrorMessage = "Структура многослойного персептрона некорректна.";
}

ETrainSetError::ETrainSetError() throw()
{
    m_sErrorMessage = "Обучающее множество некорректно.";
}

ETrainSetError::ETrainSetError(const QString& sErrorMsg) throw()
{
    m_sErrorMessage = sErrorMsg;
}

ETrainProcessError::ETrainProcessError(const QString& sIncorrectParamName)
        throw()
{
    m_sErrorMessage.append("Следующий параметр алгоритма обучения "\
                           "некорректен: ");
    m_sErrorMessage.append(sIncorrectParamName);
    m_sErrorMessage.append(".");
}

/*****************************************************************************/
/*                  РЕАЛИЗАЦИЯ КЛАССА CMultilayerPerceptron                  */
/*****************************************************************************/

/*****************************************************************************/
/* Копирование значений атрибутов другого многослойного персептрона
(private-операция, которая используется в копирующем присваивании и в операции
загрузки из файла). */
/*****************************************************************************/

void CMultilayerPerceptron::copy_from(const CMultilayerPerceptron &src)
{
    int i, nMaxLayerSize;
    size_t nDataSize;

    delete[] m_aLayerSizes;
    delete[] m_aInputsCount;
    delete[] m_aIndexesForIDBD;
    delete[] m_aWeights;
    delete[] m_aActivations;
    delete[] m_aTempOutputs;
    delete[] m_aTempInputs;

    m_nInputsCount = src.m_nInputsCount;
    m_nLayersCount = src.m_nLayersCount;

    m_aLayerSizes = new int[m_nLayersCount];
    m_aInputsCount = new int[m_nLayersCount];
    m_aIndexesForIDBD = new int[m_nLayersCount];
    m_aActivations = new TActivationKind[m_nLayersCount];

    m_aLayerSizes[0] = src.m_aLayerSizes[0];
    m_aInputsCount[0] = src.m_aInputsCount[0];
    m_aIndexesForIDBD[0] = src.m_aIndexesForIDBD[0];
    m_aActivations[0] = src.m_aActivations[0];
    nMaxLayerSize = m_aLayerSizes[0];
    for (i = 1; i < m_nLayersCount; i++)
    {
        m_aLayerSizes[i] = src.m_aLayerSizes[i];
        m_aInputsCount[i] = src.m_aInputsCount[i];
        m_aIndexesForIDBD[i] = src.m_aIndexesForIDBD[i];
        m_aActivations[i] = src.m_aActivations[i];
        if (m_aLayerSizes[i] > nMaxLayerSize)
        {
            nMaxLayerSize = m_aLayerSizes[i];
        }
    }

    m_aWeights = new float[m_aIndexesForIDBD[m_nLayersCount-1]
                            + m_aLayerSizes[m_nLayersCount-1]
                            * (m_aInputsCount[m_nLayersCount-1] + 1)];
    nDataSize = sizeof(float) * (m_aIndexesForIDBD[m_nLayersCount-1]
                + m_aLayerSizes[m_nLayersCount-1]
                * (m_aInputsCount[m_nLayersCount-1] + 1));
    memcpy(&m_aWeights[0], &(src.m_aWeights[0]), nDataSize);

    m_aTempOutputs = new float[nMaxLayerSize];
    m_aTempInputs = new float[nMaxLayerSize];
}

/*****************************************************************************/
/* Проверка корректности заданных размеров нейросети: количества входов
nInputs, количества слоёв nLayers и количества нейронов в слоях aLayerSizes[].
В случае, если значение хотя бы одного из размеров некорректно, генерируется
исключение EMLPSizeError.
   Данная операция является private-операций, используемой в конструкторе
и операции resize для проверки, правильно ли заданы размеры нейросети. */
/*****************************************************************************/
void CMultilayerPerceptron::check_size(int nInputs, int nLayers,
                                       int aLayerSizes[])
{
    if ((nInputs <= 0) || (nLayers <= 0))
    {
        throw EMLPStructError();
    }

    bool is_correct = true;
    for (int i = 0; i < nLayers; i++)
    {
        if (aLayerSizes[i] <= 0)
        {
            is_correct = false;
            break;
        }
    }
    if (!is_correct)
    {
        throw EMLPStructError();
    }
}

/*****************************************************************************/
/* Конструктор класса CMultilayerPerceptron (без аргументов - по умолчанию
создаётся нейросеть с одним входом, одним слоем и единственным нейроном в
слое). */
/*****************************************************************************/
CMultilayerPerceptron::CMultilayerPerceptron()
{
    m_nInputsCount = 1;
    m_nLayersCount = 1;
    m_aLayerSizes = new int[1];
    m_aActivations = new TActivationKind[1];

    m_aInputsCount = new int[1];
    m_aIndexesForIDBD = new int[1];
    m_aLayerSizes[0] = 1;
    m_aActivations[0] = SIG;
    m_aInputsCount[0] = m_nInputsCount;
    m_aIndexesForIDBD[0] = 0;

    m_aWeights = new float[m_aIndexesForIDBD[0] + m_aLayerSizes[0]
                           * (m_aInputsCount[0] + 1)];

    m_aTempOutputs = new float[1];
    m_aTempInputs = new float[1];

    initialize_weights();
}

/*****************************************************************************/
/* Конструктор класса CMultilayerPerceptron (создаётся нейросеть заданной
структуры). */
/*****************************************************************************/
CMultilayerPerceptron::CMultilayerPerceptron(int nInputs, int nLayers,
                                             int aLayerSizes[],
                                             TActivationKind aActivations[])
{
    check_size(nInputs, nLayers, aLayerSizes);

    m_nInputsCount = nInputs;
    m_nLayersCount = nLayers;
    m_aLayerSizes = new int[nLayers];
    m_aActivations = new TActivationKind[nLayers];

    m_aInputsCount = new int[nLayers];
    m_aIndexesForIDBD = new int[nLayers];
    m_aLayerSizes[0] = aLayerSizes[0];
    m_aActivations[0] = aActivations[0];
    m_aInputsCount[0] = m_nInputsCount;
    m_aIndexesForIDBD[0] = 0;
    int nMaxLayerSize = m_aLayerSizes[0];
    for (int i = 1; i < m_nLayersCount; i++)
    {
        m_aLayerSizes[i] = aLayerSizes[i];
        m_aActivations[i] = aActivations[i];
        m_aInputsCount[i] = m_aLayerSizes[i-1];
        m_aIndexesForIDBD[i] = m_aIndexesForIDBD[i-1]
                             + m_aLayerSizes[i-1] * (m_aInputsCount[i-1] + 1);
        if (m_aLayerSizes[i] > nMaxLayerSize)
        {
            nMaxLayerSize = m_aLayerSizes[i];
        }
    }

    m_aWeights = new float[m_aIndexesForIDBD[m_nLayersCount-1]
                           + m_aLayerSizes[m_nLayersCount-1]
                           * (m_aInputsCount[m_nLayersCount-1] + 1)];

    m_aTempOutputs = new float[nMaxLayerSize];
    m_aTempInputs = new float[nMaxLayerSize];

    initialize_weights();
}

/*****************************************************************************/
// Копирующий конструктор класса CMultilayerPerceptron
/*****************************************************************************/
CMultilayerPerceptron::CMultilayerPerceptron(const CMultilayerPerceptron &src)
{
    int i, nMaxLayerSize;
    size_t nDataSize;

    m_nInputsCount = src.m_nInputsCount;
    m_nLayersCount = src.m_nLayersCount;

    m_aLayerSizes = new int[m_nLayersCount];
    m_aActivations = new TActivationKind[m_nLayersCount];
    m_aInputsCount = new int[m_nLayersCount];
    m_aIndexesForIDBD = new int[m_nLayersCount];

    m_aLayerSizes[0] = src.m_aLayerSizes[0];
    m_aActivations[0] = src.m_aActivations[0];
    m_aInputsCount[0] = src.m_aInputsCount[0];
    m_aIndexesForIDBD[0] = src.m_aIndexesForIDBD[0];
    nMaxLayerSize = m_aLayerSizes[0];
    for (i = 1; i < m_nLayersCount; i++)
    {
        m_aLayerSizes[i] = src.m_aLayerSizes[i];
        m_aActivations[i] = src.m_aActivations[i];
        m_aInputsCount[i] = src.m_aInputsCount[i];
        m_aIndexesForIDBD[i] = src.m_aIndexesForIDBD[i];
        if (m_aLayerSizes[i] > nMaxLayerSize)
        {
            nMaxLayerSize = m_aLayerSizes[i];
        }
    }

    m_aWeights = new float[m_aIndexesForIDBD[m_nLayersCount-1]
                            + m_aLayerSizes[m_nLayersCount-1]
                            * (m_aInputsCount[m_nLayersCount-1] + 1)];
    nDataSize = sizeof(float) * (m_aIndexesForIDBD[m_nLayersCount-1]
                + m_aLayerSizes[m_nLayersCount-1]
                * (m_aInputsCount[m_nLayersCount-1] + 1));
    memcpy(&m_aWeights[0], &(src.m_aWeights[0]), nDataSize);

    m_aTempOutputs = new float[nMaxLayerSize];
    m_aTempInputs = new float[nMaxLayerSize];
}

/*****************************************************************************/
// Копирующее присваивание класса CMultilayerPerceptron
/*****************************************************************************/
CMultilayerPerceptron& CMultilayerPerceptron::operator =(
        const CMultilayerPerceptron& src)
{
    if (this != &src)
    {
        copy_from(src);
    }

    return *this;
}

/*****************************************************************************/
// Деструктор класса CMultilayerPerceptron
/*****************************************************************************/
CMultilayerPerceptron::~CMultilayerPerceptron()
{
    if (m_aLayerSizes != 0)
    {
        delete[] m_aLayerSizes;
        m_aLayerSizes = 0;
    }
    if (m_aActivations != 0)
    {
        delete[] m_aActivations;
        m_aActivations = 0;
    }
    if (m_aInputsCount != 0)
    {
        delete[] m_aInputsCount;
        m_aInputsCount = 0;
    }
    if (m_aIndexesForIDBD != 0)
    {
        delete[] m_aIndexesForIDBD;
        m_aIndexesForIDBD = 0;
    }
    if (m_aWeights != 0)
    {
        delete[] m_aWeights;
        m_aWeights = 0;
    }
    if (m_aTempOutputs != 0)
    {
        delete[] m_aTempOutputs;
        m_aTempOutputs = 0;
    }
    if (m_aTempInputs != 0)
    {
        delete[] m_aTempInputs;
        m_aTempInputs = 0;
    }
}

/*****************************************************************************/
/* Вычисление последовательности выходных сигналов inputs[] многослойного
персептрона при подаче на вход соответствующей последовательности входных
сигналов outputs[].
   Длина последовательности входных сигналов и соответствующей
последовательности вычисляемых выходных сигналов равна nSamples. */
/*****************************************************************************/
void CMultilayerPerceptron::calculate_outputs(float inputs[], float outputs[],
                                              int nSamples)
{
    if (nSamples <= 0)
    {
        throw ETrainSetError();
    }
    int iSample, iInputSampleStart, iOutputSampleStart, i, j, k;
    float sum_value;
    if (m_nLayersCount > 1) // если есть хотя бы один скрытый слой
    {
        float *pTemp;
        /* Для каждого из nSamples входных сигналов вычисляем выходной сигнал
        нейросети и записываем в соответствующее место в outputs[] */
        for (iSample = 0; iSample < nSamples; iSample++)
        {
            iInputSampleStart = iSample * m_nInputsCount;
            iOutputSampleStart = iSample * m_aLayerSizes[m_nLayersCount-1];
            /* Вычисляем выходы нейронов 1-го слоя и записываем в
            m_aTempOutputs[] */
            #pragma omp parallel for private(k,sum_value)
            for (j = 0; j < m_aLayerSizes[0]; j++)
            {
                sum_value = 0.0;
                // цикл по всем входам в нейрон
                for (k = 0; k < m_nInputsCount; k++)
                {
                    sum_value += inputs[k+iInputSampleStart]*getWeight(0,j,k);
                }
                // плюс смещение
                sum_value += getWeight(0,j,m_nInputsCount);
                // пропускаем через функцию активации
                m_aTempOutputs[j] = activation(sum_value, m_aActivations[0]);
            }

            /* Пропускаем сигнал через все скрытые слои, кроме первого
            (его выходы мы уже только что вычислили) */
            for (i = 1; i < (m_nLayersCount-1); i++)
            {
                /* Были - выходы нейронов предыдущего (i-1)-го слоя
                m_aTempOutputs[], а стали - входы нейронов текущего i-го слоя
                m_aTempInputs[].
                   Т.е. меняем местами указатели на эти массивы. */
                pTemp = m_aTempOutputs;
                m_aTempOutputs = m_aTempInputs;
                m_aTempInputs = pTemp;

                /* Вычисляем выходы нейронов i-го слоя и записываем
                m_aTempOutputs[] */
                #pragma omp parallel for private(k,sum_value)
                for (j = 0; j < m_aLayerSizes[i]; j++)
                {
                    sum_value = 0.0;
                    // цикл по всем входам в нейрон
                    for (k = 0; k < m_aInputsCount[i]; k++)
                    {
                        sum_value += m_aTempInputs[k]*getWeight(i,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(i,j,m_aInputsCount[i]);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j]=activation(sum_value,m_aActivations[i]);
                }
            }

            /* Вычисляем выходы нейронов выходного слоя, они же - выходы
            нейросети */
            i = m_nLayersCount-1;

            /* Были - выходы нейронов последнего скрытого слоя
            m_aTempOutputs[], а стали - входы нейронов выходного слоя
            m_aTempInputs[].
               Т.е. меняем местами указатели на эти массивы.
               Но для записи выходов нейронов выходного слоя используем теперь
            не m_aTempOutputs[], а уже outputs[] - массив выходов нейросети. */
            pTemp = m_aTempOutputs;
            m_aTempOutputs = m_aTempInputs;
            m_aTempInputs = pTemp;

            /* Вычисляем выходы нейронов выходного слоя и записываем в
            outputs[] */
            #pragma omp parallel for private(k,sum_value)
            for (j = 0; j < m_aLayerSizes[i]; j++)
            {
                sum_value = 0.0;
                // цикл по всем входам в нейрон
                for (k = 0; k < m_aInputsCount[i]; k++)
                {
                    sum_value += m_aTempInputs[k] * getWeight(i,j,k);
                }
                // плюс смещение
                sum_value += getWeight(i,j,m_aInputsCount[i]);
                // пропускаем через функцию активации
                outputs[j+iOutputSampleStart] = activation(sum_value,
                                                           m_aActivations[i]);
            }
        }
    }
    else  // если скрытых слоев нет, а есть только один - выходной
    {
        /* Для каждого из nSamples входных сигналов вычисляем выходной сигнал
        нейросети и записываем в соответствующее место в outputs[] */
        for (iSample = 0; iSample < nSamples; iSample++)
        {
            iInputSampleStart = iSample * m_nInputsCount;
            iOutputSampleStart = iSample * m_aLayerSizes[0];
            /* Вычисляем выходы нейронов слоя и записываем в outputs[] */
            #pragma omp parallel for private(k,sum_value)
            for (j = 0; j < m_aLayerSizes[0]; j++)
            {
                sum_value = 0.0;
                // цикл по всем входам в нейрон
                for (k = 0; k < m_nInputsCount; k++)
                {
                    sum_value += inputs[k+iInputSampleStart]*getWeight(0,j,k);
                }
                // плюс смещение
                sum_value += getWeight(0,j,m_nInputsCount);
                // пропускаем через функцию активации
                outputs[j+iOutputSampleStart] = activation(sum_value,
                                                           m_aActivations[0]);
            }
        }
    }
}

/*****************************************************************************/
/* Вычисление среднеквадратичного отклонения между последовательностью
желаемых выходных сигналов targets[] и последовательностью реальных
выходных сигналов многослойного персептрона, вычисленных при подаче
на вход соответствующей последовательности входных сигналов inputs[].
   Длина последовательности входных сигналов и соответствующей
последовательности желаемых выходных сигналов равна nSamples.
   Возвращаемое значение - вычисленное среднеквадратичное отклонение. */
/*****************************************************************************/
float CMultilayerPerceptron::calculate_mse(float inputs[], float targets[],
                                            int nSamples)
{
    if (nSamples <= 0)
    {
        throw ETrainSetError();
    }
    int iSample, iInputSampleStart, iOutputSampleStart, i, j, k;
    float result = 0.0, instant_mse, sum_value;
    if (m_nLayersCount > 1) // если есть хотя бы один скрытый слой
    {
        float *pTemp;
        /* Для каждого из nSamples входных сигналов вычисляем выходной сигнал
        нейросети и вычисляем сумму квадратов отклонений его от желаемого
        выходного сигнала из соответствующего мества outputs[] */
        for (iSample = 0; iSample < nSamples; iSample++)
        {
            iInputSampleStart = iSample * m_nInputsCount;
            iOutputSampleStart = iSample * m_aLayerSizes[m_nLayersCount-1];
            /* Вычисляем выходы нейронов 1-го слоя и записываем в
            m_aTempOutputs[] */
            #pragma omp parallel for private(k,sum_value)
            for (j = 0; j < m_aLayerSizes[0]; j++)
            {
                sum_value = 0.0;
                // цикл по всем входам в нейрон
                for (k = 0; k < m_nInputsCount; k++)
                {
                    sum_value += inputs[k+iInputSampleStart]*getWeight(0,j,k);
                }
                // плюс смещение
                sum_value += getWeight(0,j,m_nInputsCount);
                // пропускаем через функцию активации
                m_aTempOutputs[j] = activation(sum_value, m_aActivations[0]);
            }

            /* Пропускаем сигнал через все скрытые слои, кроме первого
            (его выходы мы уже только что вычислили) */
            for (i = 1; i < (m_nLayersCount-1); i++)
            {
                /* Были - выходы нейронов предыдущего (i-1)-го слоя
                m_aTempOutputs[], а стали - входы нейронов текущего i-го слоя
                m_aTempInputs[].
                   Т.е. меняем местами указатели на эти массивы. */
                pTemp = m_aTempOutputs;
                m_aTempOutputs = m_aTempInputs;
                m_aTempInputs = pTemp;

                /* Вычисляем выходы нейронов i-го слоя и записываем
                m_aTempOutputs[] */
                #pragma omp parallel for private(k,sum_value)
                for (j = 0; j < m_aLayerSizes[i]; j++)
                {
                    sum_value = 0.0;
                    // цикл по всем входам в нейрон
                    for (k = 0; k < m_aInputsCount[i]; k++)
                    {
                        sum_value += m_aTempInputs[k]*getWeight(i,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(i,j,m_aInputsCount[i]);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j]=activation(sum_value,m_aActivations[i]);
                }
            }

            /* Вычисляем выходы нейронов выходного слоя, образующие выходной
            сигнал нейросети, и записываем их в m_aTempOutputs[]. Затем сразу
            же вычисляем сумму квадратов разностей между полученным и
            соответствующим ему желаемым выходными сигналами. */
            i = m_nLayersCount-1;

            /* Были - выходы нейронов последнего скрытого слоя
            m_aTempOutputs[], а стали - входы нейронов выходного слоя
            m_aTempInputs[].
               Т.е. меняем местами указатели на эти массивы. */
            pTemp = m_aTempOutputs;
            m_aTempOutputs = m_aTempInputs;
            m_aTempInputs = pTemp;

            /* Вычисляем выходы нейронов выходного слоя и сразу же считаем
            квадратичное отклонение между полученными и желаемыми выходами. */
            instant_mse = 0.0;
            #pragma omp parallel for private(k,sum_value)\
                reduction(+:instant_mse)
            for (j = 0; j < m_aLayerSizes[i]; j++)
            {
                sum_value = 0.0;
                // цикл по всем входам в нейрон, включая вход смещения
                for (k = 0; k < m_aInputsCount[i]; k++)
                {
                    sum_value += m_aTempInputs[k] * getWeight(i,j,k);
                }
                // плюс смещение
                sum_value += getWeight(i,j,m_aInputsCount[i]);
                // пропускаем через функцию активации
                m_aTempOutputs[j] = activation(sum_value, m_aActivations[i]);

                // считаем разность между реальным и желаемым выходами
                sum_value = m_aTempOutputs[j] - targets[j+iOutputSampleStart];
                // возводим в квадрат и накапливаем
                instant_mse += (sum_value * sum_value);
            }
            instant_mse /= m_aLayerSizes[i];

            result += instant_mse;
        }
    }
    else  // если скрытых слоев нет, а есть только один - выходной
    {
        /* Для каждого из nSamples входных сигналов вычисляем выходной сигнал
        нейросети и вычисляем сумму квадратов отклонений его от желаемого
        выходного сигнала из соответствующего мества outputs[] */
        for (iSample = 0; iSample < nSamples; iSample++)
        {
            iInputSampleStart = iSample * m_nInputsCount;
            iOutputSampleStart = iSample * m_aLayerSizes[m_nLayersCount-1];

            /* Вычисляем выходы нейронов выходного слоя, образующие выходной
            сигнал нейросети, и записываем их в m_aTempOutputs[]. Затем сразу
            же вычисляем сумму квадратов разностей между полученным и
            соответствующим ему желаемым выходными сигналами. */
            instant_mse = 0.0;
            #pragma omp parallel for private(k,sum_value)\
                reduction(+:instant_mse)
            for (j = 0; j < m_aLayerSizes[0]; j++)
            {
                sum_value = 0.0;
                // цикл по всем входам в нейрон
                for (k = 0; k < m_nInputsCount; k++)
                {
                    sum_value += inputs[k+iInputSampleStart]*getWeight(0,j,k);
                }
                // плюс смещение
                sum_value += getWeight(0,j,m_nInputsCount);
                // пропускаем через функцию активации
                m_aTempOutputs[j] = activation(sum_value, m_aActivations[0]);

                // считаем разность между реальным и желаемым выходами
                sum_value = m_aTempOutputs[j] - targets[j+iOutputSampleStart];
                // возводим в квадрат и накапливаем
                instant_mse += (sum_value * sum_value);
            }
            instant_mse /= m_aLayerSizes[0];

            result += instant_mse;
        }
    }

    /* Делим вычисленное квадратичное отклонение на количество примеров и
    размер выходного сигнала, делая тем самым квадратичное отклонение
    - среднеквадратичным.
       Возвращаем результат. */
    //return (result / (nSamples * m_aLayerSizes[m_nLayersCount-1]));
    return (result / nSamples);
}

/* Вычисление среднеквадратичного отклонения между последовательностью
желаемых выходных сигналов targets[] и последовательностью реальных
выходных сигналов многослойного персептрона, вычисленных при подаче
на вход соответствующей последовательности входных сигналов inputs[].
   Длина последовательности входных сигналов и соответствующей
последовательности желаемых выходных сигналов равна nSamples.
   Распределение вероятностей последовательности входных сигналов inputs[]
задано массивом distribution[] (длина массива равна nSamples - по числу
примеров в тестовом множестве).
   Возвращаемое значение - вычисленное среднеквадратичное отклонение. */
float CMultilayerPerceptron::calculate_mse(float inputs[], float targets[],
                                           float distribution[],int nSamples)
{
    if (nSamples <= 0)
    {
        throw ETrainSetError();
    }
    int iSample, iInputSampleStart, iOutputSampleStart, i, j, k;
    float result = 0.0, sum_distribution = 0.0;
    float sample_err, sum_value;
    if (m_nLayersCount > 1) // если есть хотя бы один скрытый слой
    {
        float *pTemp;
        /* Для каждого из nSamples входных сигналов вычисляем выходной сигнал
        нейросети и вычисляем сумму квадратов отклонений его от желаемого
        выходного сигнала из соответствующего мества outputs[] */
        for (iSample = 0; iSample < nSamples; iSample++)
        {
            iInputSampleStart = iSample * m_nInputsCount;
            iOutputSampleStart = iSample * m_aLayerSizes[m_nLayersCount-1];
            /* Вычисляем выходы нейронов 1-го слоя и записываем в
            m_aTempOutputs[] */
            #pragma omp parallel for private(k,sum_value)
            for (j = 0; j < m_aLayerSizes[0]; j++)
            {
                sum_value = 0.0;
                // цикл по всем входам в нейрон
                for (k = 0; k < m_nInputsCount; k++)
                {
                    sum_value += inputs[k+iInputSampleStart]*getWeight(0,j,k);
                }
                // плюс смещение
                sum_value += getWeight(0,j,m_nInputsCount);
                // пропускаем через функцию активации
                m_aTempOutputs[j] = activation(sum_value, m_aActivations[0]);
            }

            /* Пропускаем сигнал через все скрытые слои, кроме первого
            (его выходы мы уже только что вычислили) */
            for (i = 1; i < (m_nLayersCount-1); i++)
            {
                /* Были - выходы нейронов предыдущего (i-1)-го слоя
                m_aTempOutputs[], а стали - входы нейронов текущего i-го слоя
                m_aTempInputs[].
                   Т.е. меняем местами указатели на эти массивы. */
                pTemp = m_aTempOutputs;
                m_aTempOutputs = m_aTempInputs;
                m_aTempInputs = pTemp;

                /* Вычисляем выходы нейронов i-го слоя и записываем
                m_aTempOutputs[] */
                #pragma omp parallel for private(k,sum_value)
                for (j = 0; j < m_aLayerSizes[i]; j++)
                {
                    sum_value = 0.0;
                    // цикл по всем входам в нейрон
                    for (k = 0; k < m_aInputsCount[i]; k++)
                    {
                        sum_value += m_aTempInputs[k]*getWeight(i,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(i,j,m_aInputsCount[i]);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j]=activation(sum_value,m_aActivations[i]);
                }
            }

            /* Вычисляем выходы нейронов выходного слоя, образующие выходной
            сигнал нейросети, и записываем их в m_aTempOutputs[]. Затем сразу
            же вычисляем сумму квадратов разностей между полученным и
            соответствующим ему желаемым выходными сигналами. */
            i = m_nLayersCount-1;

            /* Были - выходы нейронов последнего скрытого слоя
            m_aTempOutputs[], а стали - входы нейронов выходного слоя
            m_aTempInputs[].
               Т.е. меняем местами указатели на эти массивы. */
            pTemp = m_aTempOutputs;
            m_aTempOutputs = m_aTempInputs;
            m_aTempInputs = pTemp;

            /* Вычисляем выходы нейронов выходного слоя и сразу же считаем
            квадратичное отклонение между полученными и желаемыми выходами. */
            sample_err = 0.0;
            #pragma omp parallel for private(k,sum_value)\
                reduction(+:sample_err)
            for (j = 0; j < m_aLayerSizes[i]; j++)
            {
                sum_value = 0.0;
                // цикл по всем входам в нейрон, включая вход смещения
                for (k = 0; k <= m_aInputsCount[i]; k++)
                {
                    sum_value += m_aTempInputs[k] * getWeight(i,j,k);
                }
                // плюс смещение
                sum_value += getWeight(i,j,m_aInputsCount[i]);
                // пропускаем через функцию активации
                m_aTempOutputs[j] = activation(sum_value, m_aActivations[i]);

                // считаем разность между реальным и желаемым выходами
                sum_value = m_aTempOutputs[j] - targets[j+iOutputSampleStart];
                // возводим в квадрат и накапливаем
                sample_err += (sum_value * sum_value);
            }
            sample_err /= m_aLayerSizes[m_nLayersCount-1];
            result += sample_err * distribution[iSample];
            sum_distribution += distribution[iSample];
        }
    }
    else  // если скрытых слоев нет, а есть только один - выходной
    {
        /* Для каждого из nSamples входных сигналов вычисляем выходной сигнал
        нейросети и вычисляем сумму квадратов отклонений его от желаемого
        выходного сигнала из соответствующего мества outputs[] */
        for (iSample = 0; iSample < nSamples; iSample++)
        {
            iInputSampleStart = iSample * m_nInputsCount;
            iOutputSampleStart = iSample * m_aLayerSizes[m_nLayersCount-1];

            /* Вычисляем выходы нейронов выходного слоя, образующие выходной
            сигнал нейросети, и записываем их в m_aTempOutputs[]. Затем сразу
            же вычисляем сумму квадратов разностей между полученным и
            соответствующим ему желаемым выходными сигналами. */
            sample_err = 0.0;
            #pragma omp parallel for private(k,sum_value)\
                reduction(+:sample_err)
            for (j = 0; j < m_aLayerSizes[0]; j++)
            {
                sum_value = 0.0;
                // цикл по всем входам в нейрон
                for (k = 0; k < m_nInputsCount; k++)
                {
                    sum_value += inputs[k+iInputSampleStart]*getWeight(0,j,k);
                }
                // плюс смещение
                sum_value += getWeight(0,j,m_nInputsCount);
                // пропускаем через функцию активации
                m_aTempOutputs[j] = activation(sum_value, m_aActivations[0]);

                // считаем разность между реальным и желаемым выходами
                sum_value = m_aTempOutputs[j] - targets[j+iOutputSampleStart];
                // возводим в квадрат и накапливаем
                sample_err += (sum_value * sum_value);
            }
            sample_err /= m_aLayerSizes[m_nLayersCount-1];
            result += sample_err * distribution[iSample];
            sum_distribution += distribution[iSample];
        }
    }

    if ((sum_distribution < 0.9999) || (sum_distribution > 1.0001))
    {
        throw ETrainProcessError("the distribution of train samples");
    }

    return result;
}

/* Вычисление ошибки классификации или регресии в процентах на
последовательности входных сигналов inputs[]. Желаемые (эталонные) выходные
сигналы заданы массивом targets[].
   Длина последовательности входных сигналов и соответствующей
последовательности желаемых выходных сигналов равна nSamples.
   Возвращаемое значение - вычисленная ошибка в процентах (от 0 до 100). */
float CMultilayerPerceptron::calculate_error(float inputs[], float targets[],
                                             int nSamples, TSolvedTask task)
{
    if (nSamples <= 0)
    {
        throw ETrainSetError();
    }
    int iSample, iInputSampleStart, iOutputSampleStart, i, j, k;
    float result = 0.0, instant_error, sum_value;
    if (task == taskCLASSIFICATION) // считаем ошибку классификации
    {
        int iMaxOutput, iMaxTarget;
        if (m_aLayerSizes[m_nLayersCount-1] > 1)
        {
            if (m_nLayersCount > 1) // если есть хотя бы один скрытый слой
            {
                float *pTemp;
                /* Для каждого из nSamples входных сигналов вычисляем выходной
                сигнал нейросети и вычисляем ошибку классификации на основе его
                сравнения с желаемым выходным сигналом из соответствующего
                места outputs[] */
                for (iSample = 0; iSample < nSamples; iSample++)
                {
                    iInputSampleStart = iSample * m_nInputsCount;
                    iOutputSampleStart = iSample * m_aLayerSizes[
                            m_nLayersCount-1];
                    /* Вычисляем выходы нейронов 1-го слоя и записываем в
                    m_aTempOutputs[] */
                    #pragma omp parallel for private(k,sum_value)
                    for (j = 0; j < m_aLayerSizes[0]; j++)
                    {
                        sum_value = 0.0;
                        // цикл по всем входам в нейрон
                        for (k = 0; k < m_nInputsCount; k++)
                        {
                            sum_value += inputs[k+iInputSampleStart]
                                         * getWeight(0,j,k);
                        }
                        // плюс смещение
                        sum_value += getWeight(0,j,m_nInputsCount);
                        // пропускаем через функцию активации
                        m_aTempOutputs[j] = activation(sum_value,
                                                       m_aActivations[0]);
                    }

                    /* Пропускаем сигнал через все скрытые слои, кроме первого
                    (его выходы мы уже только что вычислили) */
                    for (i = 1; i < (m_nLayersCount-1); i++)
                    {
                        /* Были - выходы нейронов предыдущего (i-1)-го слоя
                        m_aTempOutputs[], а стали - входы нейронов текущего
                        i-го слоя m_aTempInputs[].
                           Т.е. меняем местами указатели на эти массивы. */
                        pTemp = m_aTempOutputs;
                        m_aTempOutputs = m_aTempInputs;
                        m_aTempInputs = pTemp;

                        /* Вычисляем выходы нейронов i-го слоя и записываем
                        m_aTempOutputs[] */
                        #pragma omp parallel for private(k,sum_value)
                        for (j = 0; j < m_aLayerSizes[i]; j++)
                        {
                            sum_value = 0.0;
                            // цикл по всем входам в нейрон
                            for (k = 0; k < m_aInputsCount[i]; k++)
                            {
                                sum_value += m_aTempInputs[k]*getWeight(i,j,k);
                            }
                            // плюс смещение
                            sum_value += getWeight(i,j,m_aInputsCount[i]);
                            // пропускаем через функцию активации
                            m_aTempOutputs[j] = activation(sum_value,
                                                           m_aActivations[i]);
                        }
                    }

                    /* Вычисляем выходы нейронов выходного слоя, образующие
                    выходной сигнал нейросети, и записываем их в
                    m_aTempOutputs[]. Одновременно находим номер нейрона с
                    максимальным выходом. Сравниваем этот номер с номером
                    максимального компонента желаемого выходного сигнала. Если
                    номера совпадают, то входной сигнал классифицирован
                    правильно. В противном случае фиксируется ошибка. */
                    i = m_nLayersCount-1;

                    /* Были - выходы нейронов последнего скрытого слоя
                    m_aTempOutputs[], а стали - входы нейронов выходного слоя
                    m_aTempInputs[].
                       Т.е. меняем местами указатели на эти массивы. */
                    pTemp = m_aTempOutputs;
                    m_aTempOutputs = m_aTempInputs;
                    m_aTempInputs = pTemp;

                    /* Вычисляем выходы нейронов выходного слоя и сразу же -
                    номер нейрона с максимальным выходом и номер максимального
                    компонентов желаемого выходного сигнала из соответствующего
                    места outputs[]. */
                    iMaxOutput = 0; iMaxTarget = 0;
                    for (j = 0; j < m_aLayerSizes[i]; j++)
                    {
                        sum_value = 0.0;
                        // цикл по всем входам в нейрон, включая вход смещения
                        //#pragma omp parallel for reduction(+:sum_value)
                        for (k = 0; k < m_aInputsCount[i]; k++)
                        {
                            sum_value += m_aTempInputs[k] * getWeight(i,j,k);
                        }
                        // плюс смещение
                        sum_value += getWeight(i,j,m_aInputsCount[i]);
                        // пропускаем через функцию активации
                        m_aTempOutputs[j] = activation(sum_value,
                                                       m_aActivations[i]);

                        // перепроверям номер нейрона с максимальным выходом
                        if (m_aTempOutputs[j] > m_aTempOutputs[iMaxOutput])
                        {
                            iMaxOutput = j;
                        }

                        /* перепроверям номер максимального компонента
                           желаемого выходного сигнала */
                        if (targets[j+iOutputSampleStart]
                            > targets[iMaxTarget+iOutputSampleStart])
                        {
                            iMaxTarget = j;
                        }
                    }

                    /* Если номер нейрона с максимальным выходом и номер
                    максимального компонента желаемого выходного сигнала не
                    совпадают, фиксируем ошибку классификации */
                    if (iMaxTarget != iMaxOutput)
                    {
                        result += 100.0;
                    }
                }
            }
            else  // если скрытых слоев нет, а есть только один - выходной
            {
                /* Для каждого из nSamples входных сигналов вычисляем выходной
                сигнал нейросети и вычисляем ошибку классификации на основе его
                сравнения с желаемым выходным сигналом из соответствующего
                места outputs[] */
                for (iSample = 0; iSample < nSamples; iSample++)
                {
                    iInputSampleStart = iSample * m_nInputsCount;
                    iOutputSampleStart = iSample * m_aLayerSizes[
                            m_nLayersCount-1];

                    /* Вычисляем выходы нейронов выходного слоя, образующие
                    выходной сигнал нейросети, и записываем их в
                    m_aTempOutputs[]. Затем сразу же сразу же - номер нейрона
                    с максимальным выходом и номер максимального компонентов
                    желаемого выходного сигнала из соответствующего места
                    outputs[]. */
                    iMaxOutput = 0; iMaxTarget = 0;
                    for (j = 0; j < m_aLayerSizes[0]; j++)
                    {
                        sum_value = 0.0;
                        // цикл по всем входам в нейрон
                        //#pragma omp parallel for reduction(+:sum_value)
                        for (k = 0; k < m_nInputsCount; k++)
                        {
                            sum_value += inputs[k+iInputSampleStart]
                                         * getWeight(0,j,k);
                        }
                        // плюс смещение
                        sum_value += getWeight(0,j,m_nInputsCount);
                        // пропускаем через функцию активации
                        m_aTempOutputs[j] = activation(sum_value,
                                                       m_aActivations[0]);

                        // перепроверям номер нейрона с максимальным выходом
                        if (m_aTempOutputs[j] > m_aTempOutputs[iMaxOutput])
                        {
                            iMaxOutput = j;
                        }

                        /* перепроверям номер максимального компонента
                           желаемого выходного сигнала */
                        if (targets[j+iOutputSampleStart]
                            > targets[iMaxTarget+iOutputSampleStart])
                        {
                            iMaxTarget = j;
                        }
                    }

                    /* Если номер нейрона с максимальным выходом и номер
                    максимального компонента желаемого выходного сигнала не
                    совпадают, фиксируем ошибку классификации */
                    if (iMaxTarget != iMaxOutput)
                    {
                        result += 100.0;
                    }
                }
            }
        }
        else // считаем ошибку бинарной классификации
        {
            if (m_nLayersCount > 1) // если есть хотя бы один скрытый слой
            {
                float *pTemp;
                /* Для каждого из nSamples входных сигналов вычисляем выходной
                сигнал нейросети и вычисляем ошибку классификации путём
                сравнения с желаемым выходным сигналом из соответствующего
                места outputs[] */
                for (iSample = 0; iSample < nSamples; iSample++)
                {
                    iInputSampleStart = iSample * m_nInputsCount;
                    iOutputSampleStart = iSample
                                         * m_aLayerSizes[m_nLayersCount-1];
                    /* Вычисляем выходы нейронов 1-го слоя и записываем в
                    m_aTempOutputs[] */
                    #pragma omp parallel for private(k,sum_value)
                    for (j = 0; j < m_aLayerSizes[0]; j++)
                    {
                        sum_value = 0.0;
                        // цикл по всем входам в нейрон
                        for (k = 0; k < m_nInputsCount; k++)
                        {
                            sum_value += inputs[k+iInputSampleStart]
                                         * getWeight(0,j,k);
                        }
                        // плюс смещение
                        sum_value += getWeight(0,j,m_nInputsCount);
                        // пропускаем через функцию активации
                        m_aTempOutputs[j] = activation(sum_value,
                                                       m_aActivations[0]);
                    }

                    /* Пропускаем сигнал через все скрытые слои, кроме первого
                    (его выходы мы уже только что вычислили) */
                    for (i = 1; i < (m_nLayersCount-1); i++)
                    {
                        /* Были - выходы нейронов предыдущего (i-1)-го слоя
                        m_aTempOutputs[], а стали - входы нейронов текущего
                        i-го слоя m_aTempInputs[].
                           Т.е. меняем местами указатели на эти массивы. */
                        pTemp = m_aTempOutputs;
                        m_aTempOutputs = m_aTempInputs;
                        m_aTempInputs = pTemp;

                        /* Вычисляем выходы нейронов i-го слоя и записываем
                        m_aTempOutputs[] */
                        #pragma omp parallel for private(k,sum_value)
                        for (j = 0; j < m_aLayerSizes[i]; j++)
                        {
                            sum_value = 0.0;
                            // цикл по всем входам в нейрон
                            for (k = 0; k < m_aInputsCount[i]; k++)
                            {
                                sum_value += m_aTempInputs[k]*getWeight(i,j,k);
                            }
                            // плюс смещение
                            sum_value += getWeight(i,j,m_aInputsCount[i]);
                            // пропускаем через функцию активации
                            m_aTempOutputs[j] = activation(sum_value,
                                                           m_aActivations[i]);
                        }
                    }

                    /* Вычисляем выходы нейронов выходного слоя, образующие
                    выходной сигнал нейросети, и записываем их в
                    m_aTempOutputs[]. Затем сразу же вычисляем сумму квадратов
                    разностей между полученным и соответствующим ему желаемым
                    выходными сигналами. */
                    i = m_nLayersCount-1;

                    /* Были - выходы нейронов последнего скрытого слоя
                    m_aTempOutputs[], а стали - входы нейронов выходного слоя
                    m_aTempInputs[].
                       Т.е. меняем местами указатели на эти массивы. */
                    pTemp = m_aTempOutputs;
                    m_aTempOutputs = m_aTempInputs;
                    m_aTempInputs = pTemp;

                    /* Вычисляем выход нейрона выходного слоя и сразу же ошибку
                    классификации путём сравнения с желаемым выходным сигналом
                    из соответствующего места outputs[]. */
                    j = 0.0;
                    instant_error = 0.0;

                    sum_value = 0.0;
                    // цикл по всем входам в нейрон, включая вход смещения
                    #pragma omp parallel for reduction(+:sum_value)
                    for (k = 0; k < m_aInputsCount[i]; k++)
                    {
                        sum_value += m_aTempInputs[k]*getWeight(i,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(i,j,m_aInputsCount[i]);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j] = activation(sum_value,
                                                   m_aActivations[i]);

                    /* определяем ошибку классификации на основе сравнения
                    реального и желаемого выходов */
                    if ((m_aTempOutputs[j] >= 0.0)
                        && (targets[j+iOutputSampleStart] < 0.0))
                    {
                        instant_error = 100.0;
                    }
                    else if ((m_aTempOutputs[j] < 0.0)
                        && (targets[j+iOutputSampleStart] >= 0.0))
                    {
                        instant_error = 100.0;
                    }
                    else
                    {
                        instant_error = 0.0;
                    }

                    result += instant_error;
                }
            }
            else  // если скрытых слоев нет, а есть только один - выходной
            {
                /* Для каждого из nSamples входных сигналов вычисляем выходной
                сигнал нейросети и вычисляем ошибку классификации путём
                сравнения с желаемым выходным сигналом из соответствующего
                места outputs[] */
                for (iSample = 0; iSample < nSamples; iSample++)
                {
                    iInputSampleStart = iSample * m_nInputsCount;
                    iOutputSampleStart = iSample
                                         * m_aLayerSizes[m_nLayersCount-1];

                    /* Вычисляем выход нейрона выходного слоя, образующего
                    выходной сигнал нейросети, и записываем его в
                    m_aTempOutputs[]. Затем сразу же вычисляем сумму квадратов
                    разностей между полученным и соответствующим ему желаемым
                    выходными сигналами. */
                    j = 0;
                    instant_error = 0.0;

                    sum_value = 0.0;
                    // цикл по всем входам в нейрон
                    #pragma omp parallel for reduction(+:sum_value)
                    for (k = 0; k < m_nInputsCount; k++)
                    {
                        sum_value += inputs[k+iInputSampleStart]
                                     * getWeight(0,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(0,j,m_nInputsCount);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j] = activation(sum_value,
                                                   m_aActivations[0]);

                    /* определяем ошибку классификации на основе сравнения
                    реального и желаемого выходов */
                    if ((m_aTempOutputs[j] >= 0.0)
                        && (targets[j+iOutputSampleStart] < 0.0))
                    {
                        instant_error = 100.0;
                    }
                    else if ((m_aTempOutputs[j] < 0.0)
                        && (targets[j+iOutputSampleStart] >= 0.0))
                    {
                        instant_error = 100.0;
                    }
                    else
                    {
                        instant_error = 0.0;
                    }

                    result += instant_error;
                }
            }
        }
    }
    else // считаем ошибку регрессии
    {
        if (m_nLayersCount > 1) // если есть хотя бы один скрытый слой
        {
            float *pTemp;
            /* Для каждого из nSamples входных сигналов вычисляем выходной
            сигнал нейросети и вычисляем ошибку регрессии на основе его
            отклонений от желаемого выходного сигнала из соответствующего места
            outputs[] */
            for (iSample = 0; iSample < nSamples; iSample++)
            {
                iInputSampleStart = iSample * m_nInputsCount;
                iOutputSampleStart = iSample * m_aLayerSizes[m_nLayersCount-1];
                /* Вычисляем выходы нейронов 1-го слоя и записываем в
                m_aTempOutputs[] */
                #pragma omp parallel for private(k,sum_value)
                for (j = 0; j < m_aLayerSizes[0]; j++)
                {
                    sum_value = 0.0;
                    // цикл по всем входам в нейрон
                    for (k = 0; k < m_nInputsCount; k++)
                    {
                        sum_value += inputs[k+iInputSampleStart]
                                     * getWeight(0,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(0,j,m_nInputsCount);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j] = activation(sum_value,
                                                   m_aActivations[0]);
                }

                /* Пропускаем сигнал через все скрытые слои, кроме первого
                (его выходы мы уже только что вычислили) */
                for (i = 1; i < (m_nLayersCount-1); i++)
                {
                    /* Были - выходы нейронов предыдущего (i-1)-го слоя
                    m_aTempOutputs[], а стали - входы нейронов текущего i-го
                    слоя m_aTempInputs[].
                       Т.е. меняем местами указатели на эти массивы. */
                    pTemp = m_aTempOutputs;
                    m_aTempOutputs = m_aTempInputs;
                    m_aTempInputs = pTemp;

                    /* Вычисляем выходы нейронов i-го слоя и записываем
                    m_aTempOutputs[] */
                    #pragma omp parallel for private(k,sum_value)
                    for (j = 0; j < m_aLayerSizes[i]; j++)
                    {
                        sum_value = 0.0;
                        // цикл по всем входам в нейрон
                        for (k = 0; k < m_aInputsCount[i]; k++)
                        {
                            sum_value += m_aTempInputs[k]
                                         * getWeight(i,j,k);
                        }
                        // плюс смещение
                        sum_value += getWeight(i,j,m_aInputsCount[i]);
                        // пропускаем через функцию активации
                        m_aTempOutputs[j] = activation(sum_value,
                                                       m_aActivations[i]);
                    }
                }

                /* Вычисляем выходы нейронов выходного слоя, образующие
                выходной сигнал нейросети, и записываем их в m_aTempOutputs[].
                Затем сразу же вычисляем сумму модулей разностей между
                полученным и соответствующим ему желаемым выходными сигналами.
                */
                i = m_nLayersCount-1;

                /* Были - выходы нейронов последнего скрытого слоя
                m_aTempOutputs[], а стали - входы нейронов выходного слоя
                m_aTempInputs[].
                   Т.е. меняем местами указатели на эти массивы. */
                pTemp = m_aTempOutputs;
                m_aTempOutputs = m_aTempInputs;
                m_aTempInputs = pTemp;

                /* Вычисляем выходы нейронов выходного слоя и сразу же модуль
                их среднего отклонения от компонентов желаемого выходного
                сигнала из соответствующего места outputs[]. */
                instant_error = 0.0;
                #pragma omp parallel for private(k,sum_value)\
                    reduction(+:instant_error)
                for (j = 0; j < m_aLayerSizes[i]; j++)
                {
                    sum_value = 0.0;
                    // цикл по всем входам в нейрон, включая вход смещения
                    for (k = 0; k < m_aInputsCount[i]; k++)
                    {
                        sum_value += m_aTempInputs[k]*getWeight(i,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(i,j,m_aInputsCount[i]);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j] = activation(sum_value,
                                                   m_aActivations[i]);

                    // накаплмваем ошибку регрессии
                    instant_error += regression_error(
                            m_aTempOutputs[j], targets[j+iOutputSampleStart]);
                }
                instant_error /= m_aLayerSizes[i];

                result += instant_error;
            }
        }
        else  // если скрытых слоев нет, а есть только один - выходной
        {
            /* Для каждого из nSamples входных сигналов вычисляем выходной
            сигнал нейросети и вычисляем модуль отклонения его от желаемого
            выходного сигнала из соответствующего места outputs[] */
            for (iSample = 0; iSample < nSamples; iSample++)
            {
                iInputSampleStart = iSample * m_nInputsCount;
                iOutputSampleStart = iSample * m_aLayerSizes[m_nLayersCount-1];

                /* Вычисляем выходы нейронов выходного слоя, образующие
                выходной сигнал нейросети, и записываем их в m_aTempOutputs[].
                Затем сразу же вычисляем сумму модулей разностей между
                полученным и соответствующим ему желаемым выходными сигналами.
                */
                instant_error = 0.0;
                #pragma omp parallel for private(k,sum_value)\
                    reduction(+:instant_error)
                for (j = 0; j < m_aLayerSizes[0]; j++)
                {
                    sum_value = 0.0;
                    // цикл по всем входам в нейрон
                    for (k = 0; k < m_nInputsCount; k++)
                    {
                        sum_value += inputs[k+iInputSampleStart]
                                     * getWeight(0,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(0,j,m_nInputsCount);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j] = activation(sum_value,
                                                   m_aActivations[0]);

                    instant_error += regression_error(
                            m_aTempOutputs[j], targets[j+iOutputSampleStart]);
                }
                instant_error /= m_aLayerSizes[0];

                result += instant_error;
            }
        }
    }

    /* Делим вычисленную ошибку на количество примеров. Возвращаем результат.*/
    return (result / nSamples);
}

/* Вычисление ошибки классификации или регресии в процентах на
последовательности входных сигналов inputs[]. Желаемые (эталонные) выходные
сигналы заданы массивом targets[].
   Длина последовательности входных сигналов и соответствующей
последовательности желаемых выходных сигналов равна nSamples.
   Распределение вероятностей последовательности входных сигналов inputs[]
задано массивом distribution[] (длина массива равна nSamples - по числу
примеров в тестовом множестве).
   Возвращаемое значение - вычисленная ошибка в процентах (от 0 до 100). */
float CMultilayerPerceptron::calculate_error(float inputs[], float targets[],
                                             float distribution[],
                                             int nSamples, TSolvedTask task)
{
    if (nSamples <= 0)
    {
        throw ETrainSetError();
    }
    int iSample, iInputSampleStart, iOutputSampleStart, i, j, k;
    float result = 0.0, sum_distribution = 0.0, instant_error, sum_value;
    if (task == taskCLASSIFICATION) // считаем ошибку классификации
    {
        int iMaxOutput, iMaxTarget;
        if (m_aLayerSizes[m_nLayersCount-1] > 1)
        {
            if (m_nLayersCount > 1) // если есть хотя бы один скрытый слой
            {
                float *pTemp;
                /* Для каждого из nSamples входных сигналов вычисляем выходной
                сигнал нейросети и вычисляем ошибку классификации на основе его
                сравнения с желаемым выходным сигналом из соответствующего
                места outputs[] */
                for (iSample = 0; iSample < nSamples; iSample++)
                {
                    iInputSampleStart = iSample * m_nInputsCount;
                    iOutputSampleStart = iSample * m_aLayerSizes[
                            m_nLayersCount-1];
                    /* Вычисляем выходы нейронов 1-го слоя и записываем в
                    m_aTempOutputs[] */
                    #pragma omp parallel for private(k,sum_value)
                    for (j = 0; j < m_aLayerSizes[0]; j++)
                    {
                        sum_value = 0.0;
                        // цикл по всем входам в нейрон
                        for (k = 0; k < m_nInputsCount; k++)
                        {
                            sum_value += inputs[k+iInputSampleStart]
                                         * getWeight(0,j,k);
                        }
                        // плюс смещение
                        sum_value += getWeight(0,j,m_nInputsCount);
                        // пропускаем через функцию активации
                        m_aTempOutputs[j] = activation(sum_value,
                                                       m_aActivations[0]);
                    }

                    /* Пропускаем сигнал через все скрытые слои, кроме первого
                    (его выходы мы уже только что вычислили) */
                    for (i = 1; i < (m_nLayersCount-1); i++)
                    {
                        /* Были - выходы нейронов предыдущего (i-1)-го слоя
                        m_aTempOutputs[], а стали - входы нейронов текущего
                        i-го слоя m_aTempInputs[].
                           Т.е. меняем местами указатели на эти массивы. */
                        pTemp = m_aTempOutputs;
                        m_aTempOutputs = m_aTempInputs;
                        m_aTempInputs = pTemp;

                        /* Вычисляем выходы нейронов i-го слоя и записываем
                        m_aTempOutputs[] */
                        #pragma omp parallel for private(k,sum_value)
                        for (j = 0; j < m_aLayerSizes[i]; j++)
                        {
                            sum_value = 0.0;
                            // цикл по всем входам в нейрон
                            for (k = 0; k < m_aInputsCount[i]; k++)
                            {
                                sum_value += m_aTempInputs[k]
                                             * getWeight(i,j,k);
                            }
                            // плюс смещение
                            sum_value += getWeight(i,j,m_aInputsCount[i]);
                            // пропускаем через функцию активации
                            m_aTempOutputs[j] = activation(sum_value,
                                                           m_aActivations[i]);
                        }
                    }

                    /* Вычисляем выходы нейронов выходного слоя, образующие
                    выходной сигнал нейросети, и записываем их в
                    m_aTempOutputs[]. Одновременно находим номер нейрона с
                    максимальным выходом. Сравниваем этот номер с номером
                    максимального компонента желаемого выходного сигнала. Если
                    номера совпадают, то входной сигнал классифицирован
                    правильно. В противном случае фиксируется ошибка. */
                    i = m_nLayersCount-1;

                    /* Были - выходы нейронов последнего скрытого слоя
                    m_aTempOutputs[], а стали - входы нейронов выходного слоя
                    m_aTempInputs[].
                       Т.е. меняем местами указатели на эти массивы. */
                    pTemp = m_aTempOutputs;
                    m_aTempOutputs = m_aTempInputs;
                    m_aTempInputs = pTemp;

                    /* Вычисляем выходы нейронов выходного слоя и сразу же -
                    номер нейрона с максимальным выходом и номер максимального
                    компонентов желаемого выходного сигнала из соответствующего
                    места outputs[]. */
                    iMaxOutput = 0; iMaxTarget = 0;
                    for (j = 0; j < m_aLayerSizes[i]; j++)
                    {
                        sum_value = 0.0;
                        // цикл по всем входам в нейрон, включая вход смещения
                        //#pragma omp parallel for reduction(+:sum_value)
                        for (k = 0; k < m_aInputsCount[i]; k++)
                        {
                            sum_value += m_aTempInputs[k]*getWeight(i,j,k);
                        }
                        // плюс смещение
                        sum_value += getWeight(i,j,m_aInputsCount[i]);
                        // пропускаем через функцию активации
                        m_aTempOutputs[j] = activation(sum_value,
                                                       m_aActivations[i]);

                        // перепроверям номер нейрона с максимальным выходом
                        if (m_aTempOutputs[j] > m_aTempOutputs[iMaxOutput])
                        {
                            iMaxOutput = j;
                        }

                        /* перепроверям номер максимального компонента
                           желаемого выходного сигнала */
                        if (targets[j+iOutputSampleStart]
                            > targets[iMaxTarget+iOutputSampleStart])
                        {
                            iMaxTarget = j;
                        }
                    }

                    /* Если номер нейрона с максимальным выходом и номер
                    максимального компонента желаемого выходного сигнала не
                    совпадают, фиксируем ошибку классификации */
                    if (iMaxTarget != iMaxOutput)
                    {
                        result += (100.0 * distribution[iSample]);
                    }

                    sum_distribution += distribution[iSample];
                }
            }
            else  // если скрытых слоев нет, а есть только один - выходной
            {
                /* Для каждого из nSamples входных сигналов вычисляем выходной
                сигнал нейросети и вычисляем ошибку классификации на основе его
                сравнения с желаемым выходным сигналом из соответствующего
                места outputs[] */
                for (iSample = 0; iSample < nSamples; iSample++)
                {
                    iInputSampleStart = iSample * m_nInputsCount;
                    iOutputSampleStart = iSample * m_aLayerSizes[
                            m_nLayersCount-1];

                    /* Вычисляем выходы нейронов выходного слоя, образующие
                    выходной сигнал нейросети, и записываем их в
                    m_aTempOutputs[]. Затем сразу же сразу же - номер нейрона
                    с максимальным выходом и номер максимального компонентов
                    желаемого выходного сигнала из соответствующего места
                    outputs[]. */
                    iMaxOutput = 0; iMaxTarget = 0;
                    for (j = 0; j < m_aLayerSizes[0]; j++)
                    {
                        sum_value = 0.0;
                        // цикл по всем входам в нейрон
                        //#pragma omp parallel for reduction(+:sum_value)
                        for (k = 0; k < m_nInputsCount; k++)
                        {
                            sum_value += inputs[k+iInputSampleStart]
                                         * getWeight(0,j,k);
                        }
                        // плюс смещение
                        sum_value += getWeight(0,j,m_nInputsCount);
                        // пропускаем через функцию активации
                        m_aTempOutputs[j] = activation(sum_value,
                                                       m_aActivations[0]);

                        // перепроверям номер нейрона с максимальным выходом
                        if (m_aTempOutputs[j] > m_aTempOutputs[iMaxOutput])
                        {
                            iMaxOutput = j;
                        }

                        /* перепроверям номер максимального компонента
                           желаемого выходного сигнала */
                        if (targets[j+iOutputSampleStart]
                            > targets[iMaxTarget+iOutputSampleStart])
                        {
                            iMaxTarget = j;
                        }
                    }

                    /* Если номер нейрона с максимальным выходом и номер
                    максимального компонента желаемого выходного сигнала не
                    совпадают, фиксируем ошибку классификации */
                    if (iMaxTarget != iMaxOutput)
                    {
                        result += (100.0 * distribution[iSample]);
                    }

                    sum_distribution += distribution[iSample];
                }
            }
        }
        else // считаем ошибку бинарной классификации
        {
            if (m_nLayersCount > 1) // если есть хотя бы один скрытый слой
            {
                float *pTemp;
                /* Для каждого из nSamples входных сигналов вычисляем выходной
                сигнал нейросети и вычисляем ошибку классификации путём
                сравнения с желаемым выходным сигналом из соответствующего
                места outputs[] */
                for (iSample = 0; iSample < nSamples; iSample++)
                {
                    iInputSampleStart = iSample * m_nInputsCount;
                    iOutputSampleStart = iSample
                                         * m_aLayerSizes[m_nLayersCount-1];
                    /* Вычисляем выходы нейронов 1-го слоя и записываем в
                    m_aTempOutputs[] */
                    #pragma omp parallel for private(k,sum_value)
                    for (j = 0; j < m_aLayerSizes[0]; j++)
                    {
                        sum_value = 0.0;
                        // цикл по всем входам в нейрон
                        for (k = 0; k < m_nInputsCount; k++)
                        {
                            sum_value += inputs[k+iInputSampleStart]
                                         * getWeight(0,j,k);
                        }
                        // плюс смещение
                        sum_value += getWeight(0,j,m_nInputsCount);
                        // пропускаем через функцию активации
                        m_aTempOutputs[j] = activation(sum_value,
                                                       m_aActivations[0]);
                    }

                    /* Пропускаем сигнал через все скрытые слои, кроме первого
                    (его выходы мы уже только что вычислили) */
                    for (i = 1; i < (m_nLayersCount-1); i++)
                    {
                        /* Были - выходы нейронов предыдущего (i-1)-го слоя
                        m_aTempOutputs[], а стали - входы нейронов текущего
                        i-го слоя m_aTempInputs[].
                           Т.е. меняем местами указатели на эти массивы. */
                        pTemp = m_aTempOutputs;
                        m_aTempOutputs = m_aTempInputs;
                        m_aTempInputs = pTemp;

                        /* Вычисляем выходы нейронов i-го слоя и записываем
                        m_aTempOutputs[] */
                        #pragma omp parallel for private(k,sum_value)
                        for (j = 0; j < m_aLayerSizes[i]; j++)
                        {
                            sum_value = 0.0;
                            // цикл по всем входам в нейрон
                            for (k = 0; k < m_aInputsCount[i]; k++)
                            {
                                sum_value += m_aTempInputs[k]*getWeight(i,j,k);
                            }
                            // плюс смещение
                            sum_value += getWeight(i,j,m_aInputsCount[i]);
                            // пропускаем через функцию активации
                            m_aTempOutputs[j] = activation(sum_value,
                                                           m_aActivations[i]);
                        }
                    }

                    /* Вычисляем выходы нейронов выходного слоя, образующие
                    выходной сигнал нейросети, и записываем их в
                    m_aTempOutputs[]. Затем сразу же вычисляем сумму квадратов
                    разностей между полученным и соответствующим ему желаемым
                    выходными сигналами. */
                    i = m_nLayersCount-1;

                    /* Были - выходы нейронов последнего скрытого слоя
                    m_aTempOutputs[], а стали - входы нейронов выходного слоя
                    m_aTempInputs[].
                       Т.е. меняем местами указатели на эти массивы. */
                    pTemp = m_aTempOutputs;
                    m_aTempOutputs = m_aTempInputs;
                    m_aTempInputs = pTemp;

                    /* Вычисляем выход нейрона выходного слоя и сразу же ошибку
                    классификации путём сравнения с желаемым выходным сигналом
                    из соответствующего места outputs[]. */
                    j = 0.0;
                    instant_error = 0.0;

                    sum_value = 0.0;
                    // цикл по всем входам в нейрон, включая вход смещения
                    #pragma omp parallel for reduction(+:sum_value)
                    for (k = 0; k < m_aInputsCount[i]; k++)
                    {
                        sum_value += m_aTempInputs[k] * getWeight(i,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(i,j,m_aInputsCount[i]);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j] = activation(sum_value,
                                                   m_aActivations[i]);

                    /* определяем ошибку классификации на основе сравнения
                    реального и желаемого выходов */
                    if ((m_aTempOutputs[j] >= 0.0)
                        && (targets[j+iOutputSampleStart] < 0.0))
                    {
                        instant_error = 100.0;
                    }
                    else if ((m_aTempOutputs[j] < 0.0)
                        && (targets[j+iOutputSampleStart] >= 0.0))
                    {
                        instant_error = 100.0;
                    }
                    else
                    {
                        instant_error = 0.0;
                    }

                    result += (instant_error * distribution[iSample]);
                    sum_distribution += distribution[iSample];
                }
            }
            else  // если скрытых слоев нет, а есть только один - выходной
            {
                /* Для каждого из nSamples входных сигналов вычисляем выходной
                сигнал нейросети и вычисляем ошибку классификации путём
                сравнения с желаемым выходным сигналом из соответствующего
                места outputs[] */
                for (iSample = 0; iSample < nSamples; iSample++)
                {
                    iInputSampleStart = iSample * m_nInputsCount;
                    iOutputSampleStart = iSample
                                         * m_aLayerSizes[m_nLayersCount-1];

                    /* Вычисляем выход нейрона выходного слоя, образующего
                    выходной сигнал нейросети, и записываем его в
                    m_aTempOutputs[]. Затем сразу же вычисляем сумму квадратов
                    разностей между полученным и соответствующим ему желаемым
                    выходными сигналами. */
                    j = 0;
                    instant_error = 0.0;

                    sum_value = 0.0;
                    // цикл по всем входам в нейрон
                    #pragma omp parallel for reduction(+:sum_value)
                    for (k = 0; k < m_nInputsCount; k++)
                    {
                        sum_value += inputs[k+iInputSampleStart]
                                     * getWeight(0,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(0,j,m_nInputsCount);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j] = activation(sum_value,
                                                   m_aActivations[0]);

                    /* определяем ошибку классификации на основе сравнения
                    реального и желаемого выходов */
                    if ((m_aTempOutputs[j] >= 0.0)
                        && (targets[j+iOutputSampleStart] < 0.0))
                    {
                        instant_error = 100.0;
                    }
                    else if ((m_aTempOutputs[j] < 0.0)
                        && (targets[j+iOutputSampleStart] >= 0.0))
                    {
                        instant_error = 100.0;
                    }
                    else
                    {
                        instant_error = 0.0;
                    }

                    result += (instant_error * distribution[iSample]);
                    sum_distribution += distribution[iSample];
                }
            }
        }
    }
    else // считаем ошибку регрессии
    {
        if (m_nLayersCount > 1) // если есть хотя бы один скрытый слой
        {
            float *pTemp;
            /* Для каждого из nSamples входных сигналов вычисляем выходной
            сигнал нейросети и вычисляем ошибку регрессии на основе его
            отклонений от желаемого выходного сигнала из соответствующего места
            outputs[] */
            for (iSample = 0; iSample < nSamples; iSample++)
            {
                iInputSampleStart = iSample * m_nInputsCount;
                iOutputSampleStart = iSample * m_aLayerSizes[m_nLayersCount-1];
                /* Вычисляем выходы нейронов 1-го слоя и записываем в
                m_aTempOutputs[] */
                #pragma omp parallel for private(k,sum_value)
                for (j = 0; j < m_aLayerSizes[0]; j++)
                {
                    sum_value = 0.0;
                    // цикл по всем входам в нейрон
                    for (k = 0; k < m_nInputsCount; k++)
                    {
                        sum_value += inputs[k+iInputSampleStart]
                                     * getWeight(0,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(0,j,m_nInputsCount);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j] = activation(sum_value,
                                                   m_aActivations[0]);
                }

                /* Пропускаем сигнал через все скрытые слои, кроме первого
                (его выходы мы уже только что вычислили) */
                for (i = 1; i < (m_nLayersCount-1); i++)
                {
                    /* Были - выходы нейронов предыдущего (i-1)-го слоя
                    m_aTempOutputs[], а стали - входы нейронов текущего i-го
                    слоя m_aTempInputs[].
                       Т.е. меняем местами указатели на эти массивы. */
                    pTemp = m_aTempOutputs;
                    m_aTempOutputs = m_aTempInputs;
                    m_aTempInputs = pTemp;

                    /* Вычисляем выходы нейронов i-го слоя и записываем
                    m_aTempOutputs[] */
                    #pragma omp parallel for private(k,sum_value)
                    for (j = 0; j < m_aLayerSizes[i]; j++)
                    {
                        sum_value = 0.0;
                        // цикл по всем входам в нейрон
                        for (k = 0; k < m_aInputsCount[i]; k++)
                        {
                            sum_value += m_aTempInputs[k] * getWeight(i,j,k);
                        }
                        // плюс смещение
                        sum_value += getWeight(i,j,m_aInputsCount[i]);
                        // пропускаем через функцию активации
                        m_aTempOutputs[j] = activation(sum_value,
                                                       m_aActivations[i]);
                    }
                }

                /* Вычисляем выходы нейронов выходного слоя, образующие
                выходной сигнал нейросети, и записываем их в m_aTempOutputs[].
                Затем сразу же вычисляем сумму модулей разностей между
                полученным и соответствующим ему желаемым выходными сигналами.
                */
                i = m_nLayersCount-1;

                /* Были - выходы нейронов последнего скрытого слоя
                m_aTempOutputs[], а стали - входы нейронов выходного слоя
                m_aTempInputs[].
                   Т.е. меняем местами указатели на эти массивы. */
                pTemp = m_aTempOutputs;
                m_aTempOutputs = m_aTempInputs;
                m_aTempInputs = pTemp;

                /* Вычисляем выходы нейронов выходного слоя и сразу же модуль
                их среднего отклонения от компонентов желаемого выходного
                сигнала из соответствующего места outputs[]. */
                instant_error = 0.0;
                #pragma omp parallel for private(k,sum_value)\
                    reduction(+:instant_error)
                for (j = 0; j < m_aLayerSizes[i]; j++)
                {
                    sum_value = 0.0;
                    // цикл по всем входам в нейрон, включая вход смещения
                    for (k = 0; k < m_aInputsCount[i]; k++)
                    {
                        sum_value += m_aTempInputs[k] * getWeight(i,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(i,j,m_aInputsCount[i]);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j] = activation(sum_value,
                                                   m_aActivations[i]);

                    // накаплмваем ошибку регрессии
                    instant_error += regression_error(
                            m_aTempOutputs[j], targets[j+iOutputSampleStart]);
                }
                instant_error /= m_aLayerSizes[i];

                result += (instant_error * distribution[iSample]);
                sum_distribution += distribution[iSample];
            }
        }
        else  // если скрытых слоев нет, а есть только один - выходной
        {
            /* Для каждого из nSamples входных сигналов вычисляем выходной
            сигнал нейросети и вычисляем модуль отклонения его от желаемого
            выходного сигнала из соответствующего места outputs[] */
            for (iSample = 0; iSample < nSamples; iSample++)
            {
                iInputSampleStart = iSample * m_nInputsCount;
                iOutputSampleStart = iSample * m_aLayerSizes[m_nLayersCount-1];

                /* Вычисляем выходы нейронов выходного слоя, образующие
                выходной сигнал нейросети, и записываем их в m_aTempOutputs[].
                Затем сразу же вычисляем сумму модулей разностей между
                полученным и соответствующим ему желаемым выходными сигналами.
                */
                instant_error = 0.0;
                #pragma omp parallel for private(k,sum_value)\
                    reduction(+:instant_error)
                for (j = 0; j < m_aLayerSizes[0]; j++)
                {
                    sum_value = 0.0;
                    // цикл по всем входам в нейрон
                    for (k = 0; k < m_nInputsCount; k++)
                    {
                        sum_value += inputs[k+iInputSampleStart]
                                     * getWeight(0,j,k);
                    }
                    // плюс смещение
                    sum_value += getWeight(0,j,m_nInputsCount);
                    // пропускаем через функцию активации
                    m_aTempOutputs[j] = activation(sum_value,
                                                   m_aActivations[0]);

                    // накаплмваем ошибку регрессии
                    instant_error += regression_error(
                            m_aTempOutputs[j], targets[j+iOutputSampleStart]);
                }
                instant_error /= m_aLayerSizes[0];

                /* Делим среднее отклонение по модулю на максимально возможное
                отклонение, равное 2.0, и умножаем всё на 100%. Получаем
                текущую ошибку instant_error. Накапливаем среднюю ошибку по
                всем примерам  с учётом веса */
                result += (instant_error * distribution[iSample]);
                sum_distribution += distribution[iSample];
            }
        }
    }

    if ((sum_distribution < 0.9999) || (sum_distribution > 1.0001))
    {
        throw ETrainProcessError("the distribution of train samples");
    }

    return result;
}

/*****************************************************************************/
/* Инициализация весовых коэффициентов каждого нейрона сети случайными
значениями согласно равномерному распределению с нулевым мат.ожиданием и
дисперией, равной 1.0 / sqrt(InputsCount), где InputsCount - число входов
в каждый нейрон, включая смещение (bias). */
/*****************************************************************************/
void CMultilayerPerceptron::initialize_weights()
{
    int i, j, k;
    float variance;

    /* Цикл по всем слоям */
    for (i = 0; i < m_nLayersCount; i++)
    {
        variance = 1.0 / sqrt(m_aInputsCount[i] + 1.0);
        // цикл по всем нейронам слоя
        for (j = 0; j < m_aLayerSizes[i]; j++)
        {
            /* цикл по всем входам, включая смещение:
               устанавливаем случайные значения весов нейрона в соответствии
               с равномерным распределением, имеющим нулевое мат. ожидание и
               среднеквадратичное отклонение target_variance. */
            for (k = 0; k <= m_aInputsCount[i]; k++)
            {
                setWeight(i,j,k, get_random_value(-variance, variance));
            }
        }
    }
}

/*****************************************************************************/
/* Загрузка параметров многослойного персептрона из файла. */
/*****************************************************************************/
bool CMultilayerPerceptron::load(const QString& sFilename)
{
    int *aLayerSizes = 0;
    TActivationKind *aActivations = 0;
    bool result = true;

    QFile mlp_file(sFilename);
    if (!mlp_file.open(QIODevice::ReadOnly))
    {
        result = false;
    }
    else
    {
        try
        {
            QDataStream mlp_stream(&mlp_file);
            mlp_stream.setFloatingPointPrecision(QDataStream::SinglePrecision);

            qint32 nInputsCount, nLayersCount, nTemp;
            int i;
            // Считываем размер входного сигнала и количество слоёв
            mlp_stream >> nInputsCount >> nLayersCount;
            if (mlp_stream.status() != QDataStream::Ok)
            {
                result = false;
            }
            else
            {
                /* Являются ли размер входного сигнала и количество слоёв
                положительными числами? */
                if ((nInputsCount > 0) && (nLayersCount > 0)) // если да
                {
                    aLayerSizes = new int[nLayersCount];
                    aActivations = new TActivationKind[nLayersCount];
                    /* считываем размеры каждого слоя и функции активации его
                       нейронов */
                    for (i = 0; i < nLayersCount; i++)
                    {
                        mlp_stream >> nTemp;
                        if (mlp_stream.status() == QDataStream::Ok)
                        {
                            aLayerSizes[i] = nTemp;
                            // Корректно ли указан размер слоя?
                            if (aLayerSizes[i] <= 0)
                            {
                                result = false;
                            }
                            else
                            {
                                mlp_stream >> nTemp;
                                if (mlp_stream.status() == QDataStream::Ok)
                                {
                                    /*Тип активационной функции может быть либо
                                      0 (линейная), либо 1 (сигмоида) */
                                    if ((nTemp >= 0) && (nTemp <= 1))
                                    {
                                        aActivations[i] = ((nTemp==0)?LIN:SIG);
                                    }
                                    else
                                    {
                                        result = false;
                                    }
                                }
                                else
                                {
                                    result = false;
                                }
                            }
                        }
                        else
                        {
                            result = false;
                        }
                        if (!result)
                        {
                            break;
                        }
                    }
                    /* Если получение информации о размерах слоёв прошло
                       успешно, то начинаем считывать весовые коэффициенты
                       нейронов сети */
                    if (result)
                    {
                        /* Создаём временную нейросеть, в которую будем
                           записывать прочитанные значения весовых
                           коэффициентов */
                        CMultilayerPerceptron loaded_mlp(nInputsCount,
                                                         nLayersCount,
                                                         aLayerSizes,
                                                         aActivations);
                        int i, n = loaded_mlp.getAllWeightsCount();
                        qint64 nFileSize = n * sizeof(float) + sizeof(qint32)
                                           * (2 + 2 * nLayersCount);
                        if (mlp_file.size() != nFileSize)
                        {
                            result = false;
                        }
                        else
                        {
                            for (i = 0; i < n; i++)
                            {
                                mlp_stream >> loaded_mlp.m_aWeights[i];
                                if (mlp_stream.status() != QDataStream::Ok)
                                {
                                    result = false;
                                    break;
                                }
                            }
                        }

                        /* Если чтение весовых коэффициентов прошло успешно, то
                           выполняем копирование значений атрибутов временной
                           нейросети */
                        if (result)
                        {
                            copy_from(loaded_mlp);
                        }
                    }
                }
                else
                {
                    result = false;
                }
            }

            if (aLayerSizes != 0)
            {
                delete[] aLayerSizes;
                aLayerSizes = 0;
            }
            if (aActivations != 0)
            {
                delete[] aActivations;
                aActivations = 0;
            }
            mlp_file.close();
        }
        catch(...)
        {
            if (aLayerSizes != 0)
            {
                delete[] aLayerSizes;
                aLayerSizes = 0;
            }
            if (aActivations != 0)
            {
                delete[] aActivations;
                aActivations = 0;
            }
            if (mlp_file.isOpen())
            {
                mlp_file.close();
            }
            throw;
        }
    }

    return result;
}

/*****************************************************************************/
/* Сохранение параметров многослойного персептрона в файл. */
/*****************************************************************************/
bool CMultilayerPerceptron::save(const QString& sFilename) const
{
    bool result = true;
    QFile mlp_file(sFilename);
    if (!mlp_file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        result = false;
    }
    else
    {
        try
        {
            QDataStream mlp_stream(&mlp_file);
            mlp_stream.setFloatingPointPrecision(QDataStream::SinglePrecision);

            qint32 nTemp;
            // Записываем количество входов и количество слоёв
            nTemp = m_nInputsCount;
            mlp_stream << nTemp;
            nTemp = m_nLayersCount;
            mlp_stream << nTemp;
            if (mlp_stream.status() != QDataStream::Ok)
            {
                result = false;
            }
            else
            {
                int i;
                // Для каждого слоя
                for (i = 0; i < m_nLayersCount; i++)
                {
                    // Записываем размер слоя
                    nTemp = m_aLayerSizes[i];
                    mlp_stream << nTemp;
                    if (mlp_stream.status() != QDataStream::Ok)
                    {
                        result = false;
                    }
                    if (result)
                    {
                        // Записываем тип активационной функции нейронов слоя
                        nTemp = ((m_aActivations[i] == LIN) ? 0:1);
                        mlp_stream << nTemp;
                        if (mlp_stream.status() != QDataStream::Ok)
                        {
                            result = false;
                        }
                    }
                    if (!result)
                    {
                        break;
                    }
                }
                if (result)
                {
                    int i, n = getAllWeightsCount();
                    for (i = 0; i < n; i++)
                    {
                        mlp_stream << m_aWeights[i];
                        if (mlp_stream.status() != QDataStream::Ok)
                        {
                            result = false;
                            break;
                        }
                    }
                }
            }

            mlp_file.close();
        }
        catch(...)
        {
            mlp_file.close();
            throw;
        }
    }

    return result;
}

/*****************************************************************************/
/* Изменение размеров нейросети (числа входов, числа слоёв, числа нейронов в
каждом из слоёв). */
/*****************************************************************************/
void CMultilayerPerceptron::resize(int nInputs, int nLayers, int aLayerSizes[],
                                   TActivationKind aActivations[])
{
    check_size(nInputs, nLayers, aLayerSizes);

    int i, j, nLayerSize;
    size_t nWeightsDataSize;

    /* Нам необходимо изменить размеры нейросети, не испортив при этом
    значения весовых коэффициентов, остающихся после изменений. Для этого
    выполняем следующие три шага. */

    /* 1. Создаём новую нейронную сеть заданной структуры */
    CMultilayerPerceptron new_mlp(nInputs, nLayers, aLayerSizes, aActivations);

    /* 2. Копируем весовые коэффициенты нейронов в новую нейросеть. Поскольку
    новая и старая структуры в общем случае отличаются (меньше или больше
    слоёв, разное число нейронов в одном и том же слое для старой и новой
    структур, и т.п.), то мы, естественно, копируем то, что можно.
       Например, если в i-м слое новой структуры оказалось меньше нейронов,
    чем в этом же i-м слое старой структуры, то "лишние" нейроны при
    копировании отбрасываем. */
    if (nLayers > m_nLayersCount)
    {
        nLayers = m_nLayersCount;
    }
    for (i = 0; i < nLayers; i++)
    {
        if (new_mlp.getLayerSize(i) > m_aLayerSizes[i])
        {
            nLayerSize = m_aLayerSizes[i];
        }
        else
        {
            nLayerSize = new_mlp.getLayerSize(i);
        }
        if (new_mlp.getInputsCountOfLayer(i) > m_aInputsCount[i])
        {
            nWeightsDataSize = m_aInputsCount[i] * sizeof(float);
        }
        else
        {
            nWeightsDataSize = new_mlp.getInputsCountOfLayer(i)*sizeof(float);
        }
        for (j = 0; j < nLayerSize; j++)
        {
            memcpy(&(new_mlp.m_aWeights[new_mlp.m_aIndexesForIDBD[i]
                                        + j*(new_mlp.m_aInputsCount[i]+1)]),
                   &m_aWeights[m_aIndexesForIDBD[i] + j*(m_aInputsCount[i]+1)],
                   nWeightsDataSize);
        }
    }

    /* 3. Теперь new_mlp - это результат изменения структуры существующей
    нейросети. Заменяем старую нейросеть новой. */
    copy_from(new_mlp);
}

/*****************************************************************************/
/*                      РЕАЛИЗАЦИЯ КЛАССА CTrainingOfMLP                     */
/*****************************************************************************/

/*****************************************************************************/
/* PRIVATE-ОПЕРАЦИИ КЛАССА CTrainingOfMLP */
/*****************************************************************************/

void CTrainingOfMLP::getmem_for_net_outputs()
{
    m_aNetOutputsI = new int[m_pTrainedMLP->getLayersCount()];
    int nNeuronsCount = 0;
    for (int i = 0; i < m_pTrainedMLP->getLayersCount(); i++)
    {
        m_aNetOutputsI[i] = nNeuronsCount;
        nNeuronsCount += m_pTrainedMLP->getLayerSize(i);
    }
    m_aNetOutputs = new float[nNeuronsCount];
    m_aNetOutputsD = new float[nNeuronsCount];
}

void CTrainingOfMLP::clear_data()
{
    if (m_aIndexesOfTrainInputs != 0)
    {
        delete[] m_aIndexesOfTrainInputs;
        m_aIndexesOfTrainInputs = 0;
    }
    if (m_aIndexesOfTrainTargets != 0)
    {
        delete[] m_aIndexesOfTrainTargets;
        m_aIndexesOfTrainTargets = 0;
    }
    if (m_aWeightsOfTrainSamples != 0)
    {
        delete[] m_aWeightsOfTrainSamples;
        m_aWeightsOfTrainSamples = 0;
    }
    if (m_aNetOutputs != 0)
    {
        delete[] m_aNetOutputs;
        m_aNetOutputs = 0;
    }
    if (m_aNetOutputsD != 0)
    {
        delete[] m_aNetOutputsD;
        m_aNetOutputsD = 0;
    }
    if (m_aNetOutputsI != 0)
    {
        delete[] m_aNetOutputsI;
        m_aNetOutputsI = 0;
    }
}

/*****************************************************************************/
/* PUBLIC-ОПЕРАЦИИ КЛАССА CTrainingOfMLP */
/*****************************************************************************/

/* Конструктор. По умолчанию максимально допустимое число эпох обучения
равно 100. */
CTrainingOfMLP::CTrainingOfMLP(QObject* pobj): QObject(pobj)
{
    m_nMaxEpochsCount = 100;
}

void CTrainingOfMLP::setMaxEpochsCount(int nMaxEpochsCount)
{
    if (nMaxEpochsCount < 1)
    {
        throw ETrainProcessError("the maximum epochs quantity");
    }
    m_nMaxEpochsCount = nMaxEpochsCount;
}

/* Запустить процесс обучения нейронной сети pTrainedMLP на обучающем
   множестве pTrainSet. */
void CTrainingOfMLP::train(CMultilayerPerceptron *pTrainedMLP,
                           float aTrainInputs[], float aTrainTargets[],
                           int nTrainSamples)
{
    m_aIndexesOfTrainInputs = 0;
    m_aIndexesOfTrainTargets = 0;
    m_aWeightsOfTrainSamples = 0;
    m_aNetOutputs = 0;
    m_aNetOutputsD = 0;
    m_aNetOutputsI = 0;

    try
    {
        m_pTrainedMLP = pTrainedMLP;
        m_aTrainInputs = aTrainInputs;
        m_aTrainTargets = aTrainTargets;
        m_nTrainSamples = nTrainSamples;

        m_aIndexesOfTrainInputs = new int[nTrainSamples];
        m_aIndexesOfTrainTargets = new int[nTrainSamples];

        m_aWeightsOfTrainSamples = new float[nTrainSamples];
        getmem_for_net_outputs();

        for (int i = 0; i < nTrainSamples; i++)
        {
            m_aWeightsOfTrainSamples[i] = 1.0;
            m_aIndexesOfTrainInputs[i] = i * pTrainedMLP->getInputsCount();
            m_aIndexesOfTrainTargets[i] = i * pTrainedMLP->getLayerSize(
                    pTrainedMLP->getLayersCount()-1);
        }

        m_state = tsCONTINUE;
        initialize_training();
        emit start_training();
        for (int iEpoch = 1; iEpoch <= m_nMaxEpochsCount; iEpoch++)
        {
            m_state = do_epoch(iEpoch);
            emit do_training_epoch(iEpoch);
            if (m_state != tsCONTINUE)
            {
                break;
            }
        }
        if (m_state == tsCONTINUE)
        {
            m_state = tsMAXEPOCHS;
        }
        emit end_training(m_state);
        finalize_training();

        clear_data();
    }
    catch(...)
    {
        clear_data();
        throw;
    }
}

/* Запустить процесс обучения нейронной сети pTrainedMLP на обучающем
   множестве, состоящем из входных сигналов aTrainSetInputs[] и желаемых
   выходных сигналов aTrainSetTargets[]. Распределение вероятности примеров
   задано массивом sDistribution[]. */
void CTrainingOfMLP::train(CMultilayerPerceptron *pTrainedMLP,
                           float aTrainInputs[], float aTrainTargets[],
                           float aDistribution[], int nTrainSamples)
{
    m_aIndexesOfTrainInputs = 0;
    m_aIndexesOfTrainTargets = 0;
    m_aWeightsOfTrainSamples = 0;
    m_aNetOutputs = 0;
    m_aNetOutputsD = 0;
    m_aNetOutputsI = 0;

    try
    {
        m_pTrainedMLP = pTrainedMLP;
        m_aTrainInputs = aTrainInputs;
        m_aTrainTargets = aTrainTargets;
        m_nTrainSamples = nTrainSamples;

        m_aIndexesOfTrainInputs = new int[nTrainSamples];
        m_aIndexesOfTrainTargets = new int[nTrainSamples];

        m_aWeightsOfTrainSamples = new float[nTrainSamples];
        getmem_for_net_outputs();

        float sumWeights = 0.0;
        for (int i = 0; i < nTrainSamples; i++)
        {
            m_aWeightsOfTrainSamples[i] = aDistribution[i] * nTrainSamples;
            sumWeights += m_aWeightsOfTrainSamples[i];

            m_aIndexesOfTrainInputs[i] = i * pTrainedMLP->getInputsCount();
            m_aIndexesOfTrainTargets[i] = i * pTrainedMLP->getLayerSize(
                    pTrainedMLP->getLayersCount()-1);
        }
        if (round_bond005(sumWeights) != nTrainSamples)
        {
            throw ETrainProcessError("the distribution of train samples");
        }

        m_state = tsCONTINUE;
        initialize_training();
        emit start_training();
        for (int iEpoch = 1; iEpoch <= m_nMaxEpochsCount; iEpoch++)
        {
            m_state = do_epoch(iEpoch);
            emit do_training_epoch(iEpoch);
            if (m_state != tsCONTINUE)
            {
                break;
            }
        }
        if (m_state == tsCONTINUE)
        {
            m_state = tsMAXEPOCHS;
        }
        emit end_training(m_state);
        finalize_training();

        clear_data();
    }
    catch(...)
    {
        clear_data();
        throw;
    }
}

/*****************************************************************************/
/* PROTECTED-ОПЕРАЦИИ КЛАССА CTrainingOfMLP */
/*****************************************************************************/

/* Подать на вход нейросети pNet входной сигнал из iSample-го примера
обучающего множества и вычислить выходы и производные выходов всех нейронов
этой нейросети.
   По умолчанию указатель на нейросеть pNet является нулевым. В этом случае
вычисляются выходы и производные нейронов обучаемой нейросети
m_pTrainedNet. */
void CTrainingOfMLP::calculate_outputs_and_derivatives(
        int iSample, CMultilayerPerceptron *pNet)
{
    if (pNet == 0)
    {
        pNet = m_pTrainedMLP;
    }
    int nInputsCount = pNet->getInputsCount();
    int i = 0, j, k;
    float output;

    #pragma omp parallel for private(k,output)
    for (j = 0; j < pNet->getLayerSize(i); j++)
    {
        output = pNet->getWeight(i, j, nInputsCount);
        for (k = 0; k < nInputsCount; k++)
        {
            output += (pNet->getWeight(i,j,k) * getTrainInput(iSample,k));
        }
        m_aNetOutputs[m_aNetOutputsI[i] + j] = activation(
                output, pNet->getActivationKind(i));
        m_aNetOutputsD[m_aNetOutputsI[i] + j] = activation_derivative(
                output, pNet->getActivationKind(i));
    }

    for (i = 1; i < pNet->getLayersCount(); i++)
    {
        nInputsCount = pNet->getLayerSize(i-1);
        #pragma omp parallel for private(k,output)
        for (j = 0; j < pNet->getLayerSize(i); j++)
        {
            output = pNet->getWeight(i, j, nInputsCount);
            for (k = 0; k < nInputsCount; k++)
            {
                output += (pNet->getWeight(i, j, k) * getNetOutput(i-1, k));
            }
            m_aNetOutputs[m_aNetOutputsI[i] + j] = activation(
                    output, pNet->getActivationKind(i));
            m_aNetOutputsD[m_aNetOutputsI[i] + j] = activation_derivative(
                    output, pNet->getActivationKind(i));
        }
    }
}

/* Подать на вход нейросети pNet входной сигнал из iSample-го примера
обучающего множества и вычислить выходы всех нейронов этой нейросети.
   По умолчанию указатель на нейросеть pNet является нулевым. В этом случае
вычисляются выходы нейронов обучаемой нейросети m_pTrainedNet. */
void CTrainingOfMLP::calculate_outputs(int iSample,CMultilayerPerceptron* pNet)
{
    if (pNet == 0)
    {
        pNet = m_pTrainedMLP;
    }
    int nInputsCount = pNet->getInputsCount();
    int i = 0, j, k;
    float output;

    #pragma omp parallel for private(k,output)
    for (j = 0; j < pNet->getLayerSize(i); j++)
    {
        output = pNet->getWeight(i, j, nInputsCount);
        for (k = 0; k < nInputsCount; k++)
        {
            output += (pNet->getWeight(i,j,k) * getTrainInput(iSample,k));
        }
        m_aNetOutputs[m_aNetOutputsI[i] + j] = activation(
                output, pNet->getActivationKind(i));
    }

    for (i = 1; i < pNet->getLayersCount(); i++)
    {
        nInputsCount = pNet->getLayerSize(i-1);
        #pragma omp parallel for private(k,output)
        for (j = 0; j < pNet->getLayerSize(i); j++)
        {
            output = pNet->getWeight(i, j, nInputsCount);
            for (k = 0; k < nInputsCount; k++)
            {
                output += (pNet->getWeight(i, j, k) * getNetOutput(i-1, k));
            }
            m_aNetOutputs[m_aNetOutputsI[i] + j] = activation(
                    output, pNet->getActivationKind(i));
        }
    }
}

/*****************************************************************************/
/*                 РЕАЛИЗАЦИЯ КЛАССА COnlineBackpropTraining                 */
/*****************************************************************************/

/*****************************************************************************/
/* Конструктор класса COnlineBackpropTraining */
/*****************************************************************************/
COnlineBackpropTraining::COnlineBackpropTraining(QObject* pobj)
    : CTrainingOfMLP(pobj)
{
    m_aLocalGradients1 = 0;
    m_aLocalGradients2 = 0;
    m_aBetta = 0;
    m_aH = 0;
    m_aIndexesForIDBD = 0;

    m_rate = 0.05;
    m_theta = 0.001;
    m_bAdaptiveRate = false;
}

/*****************************************************************************/
/* Деструктор класса COnlineBackpropTraining */
/*****************************************************************************/
COnlineBackpropTraining::~COnlineBackpropTraining()
{
    finalize_training();
}

/*****************************************************************************/
/* PRIVATE-ОПЕРАЦИИ КЛАССА COnlineBackpropTraining */
/*****************************************************************************/

/*****************************************************************************/
/* Установить значение свойства "НАЧАЛЬНЫЙ КОЭФФИЦИЕНТ СКОРОСТИ ОБУЧЕНИЯ" */
/*****************************************************************************/
void COnlineBackpropTraining::setStartLearningRateParam(float value)
{
   if (value <= 0.0)
    {
        throw ETrainProcessError("the initial learning rate parameter");
    }
    m_rate = value;
    m_startRate = value;
}

/*****************************************************************************/
/* Установить значение свойства "КОНЕЧНЫЙ КОЭФФИЦИЕНТ СКОРОСТИ ОБУЧЕНИЯ" */
/*****************************************************************************/
void COnlineBackpropTraining::setFinalLearningRateParam(float value)
{
   if (value <= 0.0)
    {
        throw ETrainProcessError("the final learning rate parameter");
    }
    m_finalRate = value;
}

/*****************************************************************************/
/* Установить значение свойства "ПАРАМЕТР АДАПТАЦИИ СКОРОСТИ ОБУЧЕНИЯ" */
/*****************************************************************************/
void COnlineBackpropTraining::setTheta(float value)
{
    if (value <= 0.0)
    {
        throw ETrainProcessError("the THETA (it controls adaptation of all "\
                                 "learning rate parameters)");
    }
    m_theta = value;
}

/*****************************************************************************/
/* Установить значение свойства "АДАПТИВНОСТЬ СКОРОСТИ ОБУЧЕНИЯ" */
/*****************************************************************************/
void COnlineBackpropTraining::setAdaptiveRate(bool value)
{
    m_bAdaptiveRate = value;
}

/*****************************************************************************/
/* ОБРАТНЫЙ ПРОХОД АЛГОРИТМА ОБРАТНОГО РАСПРОСТРАНЕНИЯ ОШИБКИ - ВАРИАНТ 1
   Выполнить корректировку весов сети по классическому алгоритму Online
Backprop при условии, что на сеть распространено входное воздействие
текущего примера обучающего множества. */
/*****************************************************************************/
void COnlineBackpropTraining::change_weights_by_BP(int iSample)
{
    int i, j, k, nInputsCount;
    float cur_output, cur_error, local_gradient, new_weight;
    float sample_weight = getSampleWeight(iSample);

    /* Для каждого нейрона выходного слоя:
       1) вычисляем локальный градиент (по формуле для выходного слоя);
       2) записываем его в соответствующее место массива m_aLocalGradients2[];
       3) корректируем смещение нейрона. */

    i = m_pTrainedMLP->getLayersCount()-1;
    nInputsCount = m_pTrainedMLP->getInputsCountOfLayer(i);

    #pragma omp parallel for private(cur_error,local_gradient,new_weight)
    for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
    {
        cur_error = (getTrainTarget(iSample, j) - getNetOutput(i, j))
                    * sample_weight;

        local_gradient = cur_error * getNetOutputD(i,j);

        new_weight = m_pTrainedMLP->getWeight(i,j,nInputsCount)
                     + m_rate * local_gradient;
        m_pTrainedMLP->setWeight(i,j,nInputsCount, new_weight);

        m_aLocalGradients2[j] = local_gradient;
    }

    /* Если в нейросети есть скрытые слои, то выполняем цикл от последнего
    скрытого слоя до первого */
    if (m_pTrainedMLP->getLayersCount() > 1)
    {
        float *pTemp;
        for (i = m_pTrainedMLP->getLayersCount()-2; i >= 0; i--)
        {
            nInputsCount = m_pTrainedMLP->getInputsCountOfLayer(i);

            /* Для каждого нейрона i-го слоя:
               1) вычисляем локальный градиент (по формуле для скрытого слоя);
               2) записываем его в соответствующее место массива
            m_aLocalGradients1[];
               3) корректируем веса связей между данным нейроном и всеми
            нейронами (i+1)-го слоя (теперь эти веса уже можно менять);
               4) корректируем смещение данного нейрона. */
            #pragma omp parallel for\
                private(k,cur_output,cur_error,local_gradient,new_weight)
            for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
            {
                cur_output = getNetOutput(i, j);
                cur_error = 0.0;
                for (k = 0; k < m_pTrainedMLP->getLayerSize(i+1); k++)
                {
                    cur_error += m_pTrainedMLP->getWeight(i+1,k,j)
                                 * m_aLocalGradients2[k];

                    new_weight = m_pTrainedMLP->getWeight(i+1,k,j)
                                 + m_rate * cur_output * m_aLocalGradients2[k];
                    m_pTrainedMLP->setWeight(i+1,k,j, new_weight);
                }
                local_gradient = cur_error * getNetOutputD(i, j);

                new_weight = m_pTrainedMLP->getWeight(i,j,nInputsCount)
                             + m_rate * local_gradient;
                m_pTrainedMLP->setWeight(i,j,nInputsCount, new_weight);

                m_aLocalGradients1[j] = local_gradient;
            }

            /* Сейчас массив m_aLocalGradients1[] содержит локальные градиенты
            для нейронов текущего слоя, а m_aLocalGradients2[] - локальные
            градиенты для нейронов следующего слоя.
               На следующей итерации осуществится переход на слой назад, и
            локальные градиенты нейронов текущего слоя станут локальными
            градиентами следующего слоя. Поэтому меняем местами указатели
            на массивы m_aLocalGradients1 и m_aLocalGradients2. */
            pTemp = m_aLocalGradients1;
            m_aLocalGradients1 = m_aLocalGradients2;
            m_aLocalGradients2 = pTemp;
        }
    }

    /* Сейчас в массиве m_aLocalGradients2[] содержатся локальные градиенты
    для нейронов первого слоя. Поэтому самое время скорректировать веса
    всех нейронов первого слоя, и на этом завершить обратный проход. */
    #pragma omp parallel for private(k,new_weight)
    for (j = 0; j < m_pTrainedMLP->getLayerSize(0); j++)
    {
        for (k = 0; k < nInputsCount; k++)
        {
            new_weight = m_pTrainedMLP->getWeight(0,j,k)
                         + m_rate * getTrainInput(iSample,k)
                         * m_aLocalGradients2[j];
            m_pTrainedMLP->setWeight(0,j,k, new_weight);
        }
    }
}

/*****************************************************************************/
/* ОБРАТНЫЙ ПРОХОД АЛГОРИТМА ОБРАТНОГО РАСПРОСТРАНЕНИЯ ОШИБКИ - ВАРИАНТ 1
   Выполнить корректировку весов сети по алгоритму Incremental Delta Bar Delta
при условии, что на сеть распространено входное воздействие текущего
примера обучающего множества. */
/*****************************************************************************/
void COnlineBackpropTraining::change_weights_by_IDBD(int iSample)
{
    int i, j, k, nInputsCount;
    float cur_output, cur_input, cur_error, local_gradient;
    float new_weight, delta_weight;
    float cur_rate, dBetta, newH;
    float sample_weight = getSampleWeight(iSample);

    if (m_pTrainedMLP->getLayersCount() > 1)// если есть хотя бы 1 скрытый слой
    {
        float *pTemp;

        /* Для каждого нейрона выходного слоя:
           1) вычисляем локальный градиент (по формуле для выходного слоя);
           2) записываем его в соответствующее место массива
              m_aLocalGradients2[];
           3) корректируем параметр Betta для смещения нейрона;
           4) вычисляем коэффициент скорости обучения для смещения нейрона;
           5) корректируем смещение нейрона;
           6) корректируем параметр H для смещения нейрона. */
        i = m_pTrainedMLP->getLayersCount()-1;
        nInputsCount = m_pTrainedMLP->getInputsCountOfLayer(i);

        #pragma omp parallel for private(cur_error,local_gradient,dBetta,\
            cur_rate,delta_weight,new_weight,newH)
        for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
        {
            /* вычисляем ошибку нейрона выходного слоя как разность между
               реальным и желаемым выходами */
            cur_error = (getTrainTarget(iSample, j) - getNetOutput(i, j))
                        * sample_weight;

            // вычисляем локальный градиент
            local_gradient = cur_error * getNetOutputD(i, j);

            /* вычисляем новое значение параметра Betta для смещения, и на его
               основе - значение коэффициента скорости обучения для этого же
               смещения */
            dBetta = m_theta * local_gradient * getH(i,j,nInputsCount);
            if (dBetta > 2.0)
            {
                dBetta = 2.0;
            }
            else
            {
                if (dBetta < -2.0)
                {
                    dBetta = -2.0;
                }
            }
            setBetta(i,j,nInputsCount, getBetta(i,j,nInputsCount) + dBetta);
            cur_rate = calc_exp(getBetta(i,j,nInputsCount));
            //cur_rate = m_rate;

            // корректируем смещение нейрона
            delta_weight = cur_rate * local_gradient;
            new_weight = m_pTrainedMLP->getWeight(i,j,nInputsCount)
                         + delta_weight;
            m_pTrainedMLP->setWeight(i,j,nInputsCount, new_weight);

            // вычисляем новое значение параметра H для смещения
            newH = 1.0 - cur_rate;
            if (newH <= 0.0)
            {
                newH = delta_weight;
            }
            else
            {
                newH = getH(i,j,nInputsCount) * newH + delta_weight;
            }
            setH(i,j,nInputsCount, newH);

            m_aLocalGradients2[j] = local_gradient;
        }

        // Цикл по всем скрытым слоям от последнего до первого
        for (i = m_pTrainedMLP->getLayersCount()-2; i >= 0; i--)
        {
            nInputsCount = m_pTrainedMLP->getInputsCountOfLayer(i);

            /* Для каждого нейрона i-го слоя (цикл по j):
               1) вычисляем локальный градиент (по формуле для скрытого слоя);
               2) записываем его в соответствующее место массива
            m_aLocalGradients1[];
               3) корректируем веса связей между данным нейроном и всеми
            нейронами следующего, (i+1)-го слоя (теперь эти веса уже можно
            менять):
                  3.1) корректируем параметр Betta для каждой межнейронной
                       связи;
                  3.2) вычисляем коэффициент скорости обучения для веса этой
                       межнейронной связи;
                  3.3) корректируем вес межнейронной связи;
                  3.4) корректируем параметр H для межнейронной связи;
               4) корректируем параметр Betta для смещения нейрона;
               5) вычисляем коэффициент скорости обучения для смещения нейрона;
               6) корректируем смещение данного нейрона;
               7) корректируем параметр H для смещения нейрона. */
            #pragma omp parallel for private(k,cur_output,cur_error,\
                local_gradient,dBetta,cur_rate,delta_weight,new_weight,newH)
            for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
            {
                /* с помощью обратного распространения вычисляем ошибку
                   нейрона скрытого слоя */
                cur_output = getNetOutput(i, j);
                cur_error = 0.0;
                for (k = 0; k < m_pTrainedMLP->getLayerSize(i+1); k++)
                {
                    cur_error += m_pTrainedMLP->getWeight(i+1,k,j)
                                 * m_aLocalGradients2[k];

                    /* вычисляем новое значение параметра Betta для связи между
                       j-м нейроном i-го слоя и k-м нейроном (i+1)-го слоя */
                    dBetta = m_theta * m_aLocalGradients2[k] * getH(i+1,k,j)
                             * cur_output;
                    if (dBetta > 2.0)
                    {
                        dBetta = 2.0;
                    }
                    else
                    {
                        if (dBetta < -2.0)
                        {
                            dBetta = -2.0;
                        }
                    }
                    setBetta(i+1,k,j, getBetta(i+1,k,j) + dBetta);

                    /* на основе нового значения Betta вычисляем коэффициент
                       скорости обучения */
                    cur_rate = calc_exp(getBetta(i+1,k,j));
                    //cur_rate = m_rate;

                    /* вычисляем новое значение веса связи между j-м нейроном
                       i-го слоя и k-м нейроном (i+1)-го слоя */
                    delta_weight = cur_rate*cur_output*m_aLocalGradients2[k];
                    new_weight = m_pTrainedMLP->getWeight(i+1,k,j)
                                 + delta_weight;
                    m_pTrainedMLP->setWeight(i+1,k,j, new_weight);

                    /* вычисляем новое значение параметра H для связи между
                       j-м нейроном i-го слоя и k-м нейроном (i+1)-го слоя */
                    newH = 1.0 - cur_rate * cur_output * cur_output;
                    if (newH <= 0.0)
                    {
                        newH = delta_weight;
                    }
                    else
                    {
                        newH = getH(i+1,k,j) * newH + delta_weight;
                    }
                    setH(i+1,k,j, newH);
                }

                // на основе ошибки вычисляем локальный градиент
                local_gradient = cur_error * getNetOutputD(i, j);

                /* вычисляем новое значение параметра Betta для смещения, и на
                   его основе - значение коэффициента скорости обучения для
                   этого же смещения */
                dBetta = m_theta * local_gradient * getH(i,j,nInputsCount);
                if (dBetta > 2.0)
                {
                    dBetta = 2.0;
                }
                else
                {
                    if (dBetta < -2.0)
                    {
                        dBetta = -2.0;
                    }
                }
                setBetta(i,j,nInputsCount, getBetta(i,j,nInputsCount)+dBetta);
                cur_rate = calc_exp(getBetta(i,j,nInputsCount));

                // корректируем смещение нейрона
                delta_weight = cur_rate * local_gradient;
                new_weight = m_pTrainedMLP->getWeight(i,j,nInputsCount)
                             + delta_weight;
                m_pTrainedMLP->setWeight(i,j,nInputsCount, new_weight);

                // вычисляем новое значение параметра H для смещения
                newH = 1.0 - cur_rate;
                if (newH <= 0.0)
                {
                    newH = delta_weight;
                }
                else
                {
                    newH = getH(i,j,nInputsCount) * newH + delta_weight;
                }
                setH(i,j,nInputsCount, newH);

                m_aLocalGradients1[j] = local_gradient;
            }

            /* Сейчас массив m_aLocalGradients1[] содержит локальные градиенты
            для нейронов текущего слоя, а m_aLocalGradients2[] - локальные
            градиенты для нейронов следующего слоя.
               На следующей итерации осуществится переход на слой назад, и
            локальные градиенты нейронов текущего слоя станут локальными
            градиентами следующего слоя. Поэтому меняем местами указатели
            на массивы m_aLocalGradients1 и m_aLocalGradients2. */
            pTemp = m_aLocalGradients1;
            m_aLocalGradients1 = m_aLocalGradients2;
            m_aLocalGradients2 = pTemp;
        }

        /* Сейчас в массиве m_aLocalGradients2[] содержатся локальные градиенты
        для нейронов первого слоя. Поэтому самое время скорректировать веса
        всех нейронов первого слоя, и на этом завершить обратный проход. */
        #pragma omp parallel for private(k,cur_input,dBetta,cur_rate,\
            delta_weight,new_weight,newH)
        for (j = 0; j < m_pTrainedMLP->getLayerSize(0); j++)
        {
            for (k = 0; k < nInputsCount; k++)
            {
                /* вычисляем новое значение параметра Betta для связи между
                   k-м входом и j-м нейроном 1-го слоя */
                dBetta = m_theta * m_aLocalGradients2[j] * getH(0,j,k)
                         * getTrainInput(iSample,k);
                if (dBetta > 2.0)
                {
                    dBetta = 2.0;
                }
                else
                {
                    if (dBetta < -2.0)
                    {
                        dBetta = -2.0;
                    }
                }
                setBetta(0,j,k, getBetta(0,j,k) + dBetta);

                /* на основе нового значения Betta вычисляем коэффициент
                   скорости обучения */
                cur_rate = calc_exp(getBetta(0,j,k));
                //cur_rate = m_rate;

                cur_input = getTrainInput(iSample,k);

                /* вычисляем новое значение веса связи между k-м входом и
                   j-м нейроном 1-го слоя */
                delta_weight = cur_rate * cur_input * m_aLocalGradients2[j];
                new_weight = m_pTrainedMLP->getWeight(0,j,k) + delta_weight;
                m_pTrainedMLP->setWeight(0,j,k, new_weight);

                /* вычисляем новое значение параметра H для связи между
                   k-м входом и j-м нейроном 1-го слоя */
                newH = 1.0 - cur_rate * cur_input * cur_input;
                if (newH <= 0.0)
                {
                    newH = delta_weight;
                }
                else
                {
                    newH = getH(0,j,k) * newH + delta_weight;
                }
                setH(0,j,k, newH);
            }
        }
    }
    else // если скрытых слоёв нет, а есть только выходной
    {
        nInputsCount = m_pTrainedMLP->getInputsCount();
        // Для каждого нейрона слоя (цикл по j)
        #pragma omp parallel for private(k,cur_input,cur_error,\
            local_gradient,dBetta,cur_rate,delta_weight,new_weight,newH)
        for (j = 0; j < m_pTrainedMLP->getLayerSize(0); j++)
        {
            /* вычисляем ошибку нейрона слоя как разность между реальным и
               желаемым выходами */
            cur_error = (getTrainTarget(iSample, j) - getNetOutput(0, j))
                        * sample_weight;

            // вычисляем локальный градиент
            local_gradient = cur_error * getNetOutputD(0, j);

            /* вычисляем новое значение параметра Betta для смещения, и на его
               основе - значение коэффициента скорости обучения для этого же
               смещения */
            dBetta = m_theta * local_gradient * getH(0,j,nInputsCount);
            if (dBetta > 2.0)
            {
                dBetta = 2.0;
            }
            else
            {
                if (dBetta < -2.0)
                {
                    dBetta = -2.0;
                }
            }
            setBetta(0,j,nInputsCount, getBetta(0,j,nInputsCount) + dBetta);
            cur_rate = calc_exp(getBetta(0,j,nInputsCount));

            // корректируем смещение нейрона
            delta_weight = cur_rate * local_gradient;
            new_weight = m_pTrainedMLP->getWeight(0,j,nInputsCount)
                         + delta_weight;
            m_pTrainedMLP->setWeight(0,j,nInputsCount, new_weight);

            // вычисляем новое значение параметра H для смещения
            newH = 1.0 - cur_rate;
            if (newH <= 0.0)
            {
                newH = delta_weight;
            }
            else
            {
                newH = getH(0,j,nInputsCount) * newH + delta_weight;
            }
            setH(0,j,nInputsCount, newH);

            m_aLocalGradients2[j] = local_gradient;

            // Для всех входов j-го нейрона (цикл по k)
            for (k = 0; k < nInputsCount; k++)
            {
                /* вычисляем новое значение параметра Betta для связи между
                   k-м входом и j-м нейроном  */
                dBetta = m_theta * local_gradient * getH(0,j,k)
                         * getTrainInput(iSample,k);
                if (dBetta > 2.0)
                {
                    dBetta = 2.0;
                }
                else
                {
                    if (dBetta < -2.0)
                    {
                        dBetta = -2.0;
                    }
                }
                setBetta(0,j,k, getBetta(0,j,k) + dBetta);

                /* на основе нового значения Betta вычисляем коэффициент
                   скорости обучения */
                cur_rate = calc_exp(getBetta(0,j,k));

                cur_input = getTrainInput(iSample, k);

                /* вычисляем новое значение веса связи между k-м входом и
                   j-м нейроном */
                delta_weight = cur_rate * cur_input * local_gradient;
                new_weight = m_pTrainedMLP->getWeight(0,j,k) + delta_weight;
                m_pTrainedMLP->setWeight(0,j,k, new_weight);

                /* вычисляем новое значение параметра H для связи между
                   k-м входом и j-м нейроном */
                newH = 1.0 - cur_rate * cur_input * cur_input;
                if (newH <= 0.0)
                {
                    newH = delta_weight;
                }
                else
                {
                    newH = getH(0,j,k) * newH + delta_weight;
                }
                setH(0,j,k, newH);
            }
        }
    }
}

/* Инициализировать значения параметров "Betta" и "H", определяющих процесс
адаптации коэффициентов скорости обучения для каждого весового коэффициента
нейронной сети.
   Правила инициализации таковы.
   1. Начальное значение свойства "H" для каждого весового коэффициента
нейронной сети всегда равно нулю.
   2. Начальное значение свойства "Betta" для каждого весового коэффициента
нейронной сети подбирается индивидуально. Сначала определяется оптимальный
начальный коэффициент скорости обучения rate для этого весового
коэффициента, исходя из того, в каком слое находится нейрон, к которому
относится весовой коэффициент, и сколько входов этот нейрон имеет. Затем
вычисляется betta = log(rate). */
void COnlineBackpropTraining::initialize_Betta_and_H()
{
    int nAllWeightsCount = m_pTrainedMLP->getAllWeightsCount();
    float mean_inputs_ratio = 0.0, instant_inputs_ratio;
    float layers_ratio = 1.0;
    float current_learning_rate, current_betta;
    int iLayer, iWeight = 0;
    int nInputsCount, nLayerSize, nWeightsOfLayer;
    int iCounter;
    for (iLayer = 0; iLayer < m_pTrainedMLP->getLayersCount(); iLayer++)
    {
        mean_inputs_ratio += (m_pTrainedMLP->getInputsCountOfLayer(iLayer) + 1);
        layers_ratio *= 1.2;
    }
    mean_inputs_ratio /= ((float)m_pTrainedMLP->getLayersCount());
    mean_inputs_ratio= sqrt(mean_inputs_ratio);

    m_aBetta = new float[nAllWeightsCount];
    m_aH = new float[nAllWeightsCount];
    nInputsCount = m_pTrainedMLP->getInputsCount();
    for (iLayer = 0; iLayer < m_pTrainedMLP->getLayersCount(); iLayer++)
    {
        nLayerSize = m_pTrainedMLP->getLayerSize(iLayer);

        layers_ratio /= 1.2;
        instant_inputs_ratio = sqrt(nInputsCount + 1.0);
        current_learning_rate = (m_rate * mean_inputs_ratio * layers_ratio)
                / instant_inputs_ratio;
        current_betta = log(current_learning_rate);

        nWeightsOfLayer = nLayerSize * (nInputsCount + 1);
        for (iCounter = 1; iCounter <= nWeightsOfLayer; iCounter++)
        {
            m_aH[iWeight] = 0.0;
            m_aBetta[iWeight] = current_betta;
            iWeight++;
        }

        nInputsCount = nLayerSize;
    }
}

/*****************************************************************************/
/* PROTECTED-ОПЕРАЦИИ КЛАССА COnlineBackpropTraining */
/*****************************************************************************/

/*****************************************************************************/
// Выделить память для всех промежуточных переменных и инициализировать их
/*****************************************************************************/
void COnlineBackpropTraining::initialize_training()
{
    CTrainingOfMLP::initialize_training();

    int i, nMaxLayerSize;

    nMaxLayerSize = m_pTrainedMLP->getLayerSize(0);
    for (i = 1; i < m_pTrainedMLP->getLayersCount(); i++)
    {
        if (m_pTrainedMLP->getLayerSize(i) > nMaxLayerSize)
        {
            nMaxLayerSize = m_pTrainedMLP->getLayerSize(i);
        }
    }
    m_aLocalGradients1 = new float[nMaxLayerSize];
    m_aLocalGradients2 = new float[nMaxLayerSize];
    if (m_bAdaptiveRate)
    {
        initialize_Betta_and_H();

        int nWeightsCountOfLayer = m_pTrainedMLP->getLayerSize(0)
                                   * (m_pTrainedMLP->getInputsCount() + 1);
        m_aIndexesForIDBD = new int[m_pTrainedMLP->getLayersCount()];
        m_aIndexesForIDBD[0] = 0;
        for (i = 1; i < m_pTrainedMLP->getLayersCount(); i++)
        {
            m_aIndexesForIDBD[i] = m_aIndexesForIDBD[i-1] + nWeightsCountOfLayer;
            nWeightsCountOfLayer = m_pTrainedMLP->getLayerSize(i)
                                   * (m_pTrainedMLP->getInputsCountOfLayer(i)
                                      + 1);
        }
    }

    m_aIndexesOfTrainSamples = new int[getNumberOfTrainSamples()];
    calculate_rand_indexes(m_aIndexesOfTrainSamples,getNumberOfTrainSamples());
}

/*****************************************************************************/
// Освободить память ото всех промежуточных переменных
/*****************************************************************************/
void COnlineBackpropTraining::finalize_training()
{
    if (m_aLocalGradients1 != 0)
    {
        delete[] m_aLocalGradients1;
        m_aLocalGradients1 = 0;
    }
    if (m_aLocalGradients2 != 0)
    {
        delete[] m_aLocalGradients2;
        m_aLocalGradients2 = 0;
    }
    if (m_aBetta != 0)
    {
        delete[] m_aBetta;
        m_aBetta = 0;
    }
    if (m_aH != 0)
    {
        delete[] m_aH;
        m_aH = 0;
    }
    if (m_aIndexesForIDBD != 0)
    {
        delete[] m_aIndexesForIDBD;
        m_aIndexesForIDBD = 0;
    }
    if (m_aIndexesOfTrainSamples != 0)
    {
        delete[] m_aIndexesOfTrainSamples;
        m_aIndexesOfTrainSamples = 0;
    }
    CTrainingOfMLP::finalize_training();
}

/*****************************************************************************/
/* Выполнить очередную nEpoch-ю эпоху обучения */
/*****************************************************************************/

TTrainingState COnlineBackpropTraining::do_epoch(int nEpoch)
{
    int i;
    calculate_rand_indexes(m_aIndexesOfTrainSamples,getNumberOfTrainSamples());
    if (m_bAdaptiveRate)
    {
        for (i = 0; i < getNumberOfTrainSamples(); i++)
        {
            calculate_outputs_and_derivatives(m_aIndexesOfTrainSamples[i]);
            change_weights_by_IDBD(m_aIndexesOfTrainSamples[i]);
        }
    }
    else
    {
        m_rate = m_startRate + (nEpoch - 1.0) * (m_finalRate - m_startRate)
                / (getMaxEpochsCount() - 1.0);
        for (i = 0; i < getNumberOfTrainSamples(); i++)
        {
            calculate_outputs_and_derivatives(m_aIndexesOfTrainSamples[i]);
            change_weights_by_BP(m_aIndexesOfTrainSamples[i]);
        }
    }
    return tsCONTINUE;
}

/*****************************************************************************/
/*                  РЕАЛИЗАЦИЯ КЛАССА CBatchBackpropTraining                 */
/*****************************************************************************/

/*****************************************************************************/
/* PRIVATE-ОПЕРАЦИИ КЛАССА CBatchBackpropTraining */
/*****************************************************************************/

/*****************************************************************************/
/* ОБРАТНЫЙ ПРОХОД АЛГОРИТМА ОБРАТНОГО РАСПРОСТРАНЕНИЯ ОШИБКИ
   Вычислить текущий градиент по всем весам и смещениям обучаемой нейросети при
условии, что на сеть распространено входное воздействие текущего примера
обучающего множества. */
/*****************************************************************************/
void CBatchBackpropTraining::calculate_cur_gradient(int iSample)
{
    int i, j, k, nInputsCount;
    float cur_output, cur_error, mean_error = 0.0;

    /* Для каждого нейрона выходного слоя:
       1) вычисляем локальный градиент (по формуле для выходного слоя);
       2) записываем его в соответствующее место массива m_aLocalGradients2[];
       3) обновляем вектор текущего градиента m_aCurG. */
    i = m_pTrainedMLP->getLayersCount()-1;
    nInputsCount = m_pTrainedMLP->getInputsCountOfLayer(i);

    #pragma omp parallel for private(cur_output,cur_error)\
        reduction(+:mean_error)
    for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
    {
        cur_output = getNetOutput(i, j);
        cur_error = getTrainTarget(iSample, j) - cur_output;

        mean_error += (cur_error * cur_error);

        m_aLocalGradients2[j] = cur_error * getNetOutputD(i, j);

        setCurGradient(i, j, nInputsCount, m_aLocalGradients2[j]);
    }
    mean_error /= m_pTrainedMLP->getLayerSize(i);
    m_meanError += (mean_error * getSampleWeight(iSample));

    /* Если в нейросети есть скрытые слои, то выполняем цикл от последнего
    скрытого слоя до первого */
    if (m_pTrainedMLP->getLayersCount() > 1)
    {
        float *pTemp;
        for (i = m_pTrainedMLP->getLayersCount()-2; i >= 0; i--)
        {
            nInputsCount = m_pTrainedMLP->getInputsCountOfLayer(i);

            /* Для каждого нейрона i-го слоя:
               1) вычисляем локальный градиент (по формуле для скрытого слоя);
               2) записываем его в соответствующее место массива
            m_aLocalGradients1[];
               3) обновляем вектор текущего градиента m_aCurG. */
            #pragma omp parallel for private(k,cur_output,cur_error)
            for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
            {
                cur_output = getNetOutput(i, j);
                cur_error = 0.0;
                for (k = 0; k < m_pTrainedMLP->getLayerSize(i+1); k++)
                {
                    cur_error += m_pTrainedMLP->getWeight(i+1,k,j)
                                 * m_aLocalGradients2[k];

                    setCurGradient(i+1,k,j, cur_output*m_aLocalGradients2[k]);
                }
                m_aLocalGradients1[j] = cur_error * getNetOutputD(i, j);

                setCurGradient(i, j, nInputsCount, m_aLocalGradients1[j]);
            }

            /* Сейчас массив m_aLocalGradients1[] содержит локальные градиенты
            для нейронов текущего слоя, а m_aLocalGradients2[] - локальные
            градиенты для нейронов следующего слоя.
               На следующей итерации осуществится переход на слой назад, и
            локальные градиенты нейронов текущего слоя станут локальными
            градиентами следующего слоя. Поэтому меняем местами указатели
            на массивы m_aLocalGradients1 и m_aLocalGradients2. */
            pTemp = m_aLocalGradients1;
            m_aLocalGradients1 = m_aLocalGradients2;
            m_aLocalGradients2 = pTemp;
        }
    }

    /* Сейчас в массиве m_aLocalGradients2[] содержатся локальные градиенты
    для нейронов первого слоя. Поэтому самое время скорректировать локальные
    градиенты всех весов всех нейронов первого слоя, и на этом завершить
    вычисление текущего градиента обучаемой нейросети. */
    #pragma omp parallel for private(k)
    for (j = 0; j < m_pTrainedMLP->getLayerSize(0); j++)
    {
        for (k = 0; k < nInputsCount; k++)
        {
            setCurGradient(0, j, k, getTrainInput(iSample, k)
                           * m_aLocalGradients2[j]);
        }
    }
}

/*****************************************************************************/
/* PUBLIC-ОПЕРАЦИИ КЛАССА CBatchBackpropTraining */
/*****************************************************************************/

/*****************************************************************************/
/* Конструктор класса CBatchBackpropTraining */
/*****************************************************************************/

CBatchBackpropTraining::CBatchBackpropTraining(QObject* pobj)
    : CTrainingOfMLP(pobj)
{
    m_aLocalGradients1 = 0;
    m_aLocalGradients2 = 0;
    m_aCurG = 0;
    m_aMeanG = 0;
    m_aGradientIndexes = 0;
    m_epsilon = 0.0001;
}

/*****************************************************************************/
/* Деструктор класса CBatchBackpropTraining */
/*****************************************************************************/
CBatchBackpropTraining::~CBatchBackpropTraining()
{
    finalize_training();
}

void CBatchBackpropTraining::setEpsilon(float value)
{
    if ((value < 0.0) || (value >= 1.0))
    {
        throw ETrainProcessError("the EPSILON (threshold of the gradient "\
                                 "norm decrease)");
    }
    m_epsilon = value;
}

/*****************************************************************************/
/* PROTECTED-ОПЕРАЦИИ КЛАССА CBatchBackpropTraining */
/*****************************************************************************/

/*****************************************************************************/
// Выделить память для всех промежуточных переменных и инициализировать их
/*****************************************************************************/
void CBatchBackpropTraining::initialize_training()
{
    CTrainingOfMLP::initialize_training();

    int i, nMaxLayerSize = m_pTrainedMLP->getLayerSize(0);
    for (i = 1; i < m_pTrainedMLP->getLayersCount(); i++)
    {
        if (m_pTrainedMLP->getLayerSize(i) > nMaxLayerSize)
        {
            nMaxLayerSize = m_pTrainedMLP->getLayerSize(i);
        }
    }
    m_aLocalGradients1 = new float[nMaxLayerSize];
    m_aLocalGradients2 = new float[nMaxLayerSize];

    m_aCurG = new float[m_pTrainedMLP->getAllWeightsCount()];
    m_aMeanG = new float[m_pTrainedMLP->getAllWeightsCount()];

    int nWeightsCountOfLayer = m_pTrainedMLP->getLayerSize(0)
                               * (m_pTrainedMLP->getInputsCount() + 1);
    m_aGradientIndexes = new int[m_pTrainedMLP->getLayersCount()];
    m_aGradientIndexes[0] = 0;
    for (i = 1; i < m_pTrainedMLP->getLayersCount(); i++)
    {
        m_aGradientIndexes[i] = m_aGradientIndexes[i-1] + nWeightsCountOfLayer;
        nWeightsCountOfLayer = m_pTrainedMLP->getLayerSize(i)
                               * (m_pTrainedMLP->getInputsCountOfLayer(i)
                                  + 1);
    }
}

/*****************************************************************************/
// Освободить память ото всех промежуточных переменных
/*****************************************************************************/
void CBatchBackpropTraining::finalize_training()
{
    if (m_aLocalGradients1 != 0)
    {
        delete[] m_aLocalGradients1;
        m_aLocalGradients1 = 0;
    }
    if (m_aLocalGradients2 != 0)
    {
        delete[] m_aLocalGradients2;
        m_aLocalGradients2 = 0;
    }
    if (m_aCurG != 0)
    {
        delete[] m_aCurG;
        m_aCurG = 0;
    }
    if (m_aMeanG != 0)
    {
        delete[] m_aMeanG;
        m_aMeanG = 0;
    }
    if (m_aGradientIndexes != 0)
    {
        delete[] m_aGradientIndexes;
        m_aGradientIndexes = 0;
    }
    CTrainingOfMLP::finalize_training();
}

/*****************************************************************************/
/* Выполнить очередную nEpoch-ю эпоху обучения */
/*****************************************************************************/

TTrainingState CBatchBackpropTraining::do_epoch(int nEpoch)
{
    m_meanError = 0.0;
    int i, j;

    calculate_outputs_and_derivatives(0, m_pTrainedMLP);
    calculate_cur_gradient(0);
    for (j = 0; j < m_pTrainedMLP->getAllWeightsCount(); j++)
    {
        m_aMeanG[j] = m_aCurG[j] * getSampleWeight(0);
    }

    for (i = 1; i < getNumberOfTrainSamples(); i++)
    {
        calculate_outputs_and_derivatives(i, m_pTrainedMLP);
        calculate_cur_gradient(i);
        for (j = 0; j < m_pTrainedMLP->getAllWeightsCount(); j++)
        {
            m_aMeanG[j] += (m_aCurG[j] * getSampleWeight(i));
        }
    }
    m_meanError /= getNumberOfTrainSamples();

    m_meanGradientNorm = 0.0;
    for (j = 0; j < m_pTrainedMLP->getAllWeightsCount(); j++)
    {
        m_aMeanG[j] /= getNumberOfTrainSamples();
        m_meanGradientNorm += (m_aMeanG[j] * m_aMeanG[j]);
    }
    m_meanGradientNorm = sqrt(m_meanGradientNorm);

    TTrainingState result = tsCONTINUE;
    if (nEpoch > 1)
    {
        if (m_meanGradientNorm > 0.0)
        {
            if (m_meanGradientNorm < (m_epsilon * m_initGradientNorm))
            {
                result = tsGRADIENT;
            }
        }
        else
        {
            result = tsGRADIENT;
        }
    }
    else
    {
        m_initGradientNorm = m_meanGradientNorm;
        if (m_initGradientNorm <= 0.0)
        {
            result = tsGRADIENT;
        }
    }
    if (result == tsCONTINUE)
    {
        change_weights(nEpoch, result);
    }
    return result;
}

/*****************************************************************************/
/*                РЕАЛИЗАЦИЯ КЛАССА CResilientBackpropTraining               */
/*****************************************************************************/

/*****************************************************************************/
// Выделить память для всех промежуточных переменных и инициализировать их
/*****************************************************************************/
void CResilientBackpropTraining::initialize_training()
{
    CBatchBackpropTraining::initialize_training();

    int nAllWeightsCount = m_pTrainedMLP->getAllWeightsCount();

    m_aRates = new float[nAllWeightsCount];
    m_aPrevG = new float[nAllWeightsCount];
    for (int i = 0; i < nAllWeightsCount; i++)
    {
        m_aPrevG[i] = 0.0;
        m_aRates[i] = m_initLearningRate;
    }
}

/*****************************************************************************/
// Освободить память ото всех промежуточных переменных
/*****************************************************************************/
void CResilientBackpropTraining::finalize_training()
{
    if (m_aRates != 0)
    {
        delete[] m_aRates;
        m_aRates = 0;
    }
    if (m_aPrevG != 0)
    {
        delete[] m_aPrevG;
        m_aPrevG = 0;
    }
    CBatchBackpropTraining::finalize_training();
}

/*****************************************************************************/
/* Установить новое значение свойства "НАЧАЛЬНОЕ ЗНАЧЕНИЕ СКОРОСТИ ОБУЧЕНИЯ" */
/*****************************************************************************/
void CResilientBackpropTraining::setInitialLearningRate(float learning_rate)
{
    if (learning_rate <= 0.0)
    {
        throw ETrainProcessError("the initial learning rate parameter");
    }
    m_initLearningRate = learning_rate;
}

/*****************************************************************************/
/* Конструктор класса CResilientBackpropTraining */
/*****************************************************************************/
CResilientBackpropTraining::CResilientBackpropTraining(QObject* pobj)
    : CBatchBackpropTraining(pobj)
{
    m_aRates = 0;
    m_aPrevG = 0;

    m_initLearningRate = 0.01;

    m_minLearningRate = 1e-6;
    m_maxLearningRate = 50.0;
}

/*****************************************************************************/
/* Деструктор класса CResilientBackpropTraining */
/*****************************************************************************/
CResilientBackpropTraining::~CResilientBackpropTraining()
{
    finalize_training();
}

void CResilientBackpropTraining::change_weights(int nEpoch,
                                                TTrainingState& training_state)
{
    int i, j, k, n, iStartWeight = 0, iWeight;
    float temp;
    float new_weight, new_rate;
    for (i = 0; i < m_pTrainedMLP->getLayersCount(); i++)
    {
        n = m_pTrainedMLP->getInputsCountOfLayer(i);
        #pragma omp parallel for private(k,temp,new_rate,iWeight,new_weight)
        for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
        {
            iWeight = iStartWeight + j * (n+1);
            for (k = 0; k <= n; k++)
            {
                temp = getMeanGradient(i, j, k) * m_aPrevG[iWeight];
                if (temp > 0.0)
                {
                    new_rate = 1.2 * m_aRates[iWeight];
                    if (new_rate > m_maxLearningRate)
                    {
                        new_rate = m_maxLearningRate;
                    }
                }
                else if (temp < 0.0)
                {
                    new_rate = 0.5 * m_aRates[iWeight];
                    if (new_rate < m_minLearningRate)
                    {
                        new_rate = m_minLearningRate;
                    }
                }
                else
                {
                    new_rate = m_aRates[iWeight];
                }
                new_weight = m_pTrainedMLP->getWeight(i, j, k)
                             + new_rate * sign(getMeanGradient(i, j, k));
                m_pTrainedMLP->setWeight(i, j, k, new_weight);

                m_aRates[iWeight] = new_rate;
                m_aPrevG[iWeight] = getMeanGradient(i, j, k);

                iWeight++;
            }
        }
        iStartWeight += (m_pTrainedMLP->getLayerSize(i) * (n+1));
    }
}

/*****************************************************************************/
/*                РЕАЛИЗАЦИЯ КЛАССА CGradientDescentTraining               */
/*****************************************************************************/

/*****************************************************************************/
/* PRIVATE-ОПЕРАЦИИ КЛАССА CGradientDescentTraining */
/*****************************************************************************/

/*****************************************************************************/
/* Установить новое значение свойства "МАКСИМАЛЬНАЯ СКОРОСТЬ ОБУЧЕНИЯ". */
/*****************************************************************************/
void CGradientDescentTraining::setMaxLearningRate(float value)
{
    if (value <= 0.0)
    {
        throw ETrainProcessError("the maximum learning rate parameter");
    }
    m_maxLearningRate = value;
}

void CGradientDescentTraining::setMaxItersForLR(int value)
{
    if (value <= 0)
    {
        throw ETrainProcessError("the maximum iterations number for the "\
                                 "learning rate parameter search");
    }
    m_nMaxItersForLR = value;
}

/* Найти три начальные точки (lr1; etr1), (lr2; etr2) и (lr3; etr3), на основе
которых будет осуществлятся поиск оптимального шага lr в заданном направлении
m_aDirection[] (критерием оптимальности выступает ошибка обучения etr, которую
надо минимизировать). */
bool CGradientDescentTraining::find_init_lrs(
        float& lr1,float& lr2,float& lr3, float& etr1,float& etr2,float& etr3)
{
    float ulim, u, r, q, fu, tiny = FLT_EPSILON;

    lr1 = 0.0;
    lr3 = m_maxLearningRate;
    lr2 = lr3 / (1.0 + GOLD);

    etr1 = getMeanError();
    etr2 = calculate_training_error(lr2);
    while ((etr2 > etr1) && (lr2 > (lr1 + tiny)))
    {
        lr3 = lr2;
        lr2 = lr3 / (1.0 + GOLD);
        etr2 = calculate_training_error(lr2);
    }
    if (lr2 <= (lr1 + tiny))
    {
        return false;
    }
    etr3 = calculate_training_error(lr3);

    while (lr2 > lr3)
    {
        r = (lr2 - lr1) * (etr2 - etr3);
        q = (lr2 - lr3) * (etr2 - etr1);
        u = lr2 - ((lr2 - lr3) * q - (lr2 - lr1) * r)
            / (2.0 * SIGN(max(fabs(q - r), tiny), q - r));
        ulim = lr2 + 100.0 * (lr3 - lr2);
        if (((lr2 - u) * (u - lr3)) > 0.0)
        {
            fu = calculate_training_error(u);
            if (fu < etr3)
            {
                lr1 = lr2;
                lr2 = u;
                etr1 = etr2;
                etr2 = fu;
                return true;
            }
            else
            {
                if (fu > etr2)
                {
                    lr3 = u;
                    etr3 = fu;
                    return true;
                }
            }
            u = lr3 + GOLD * (lr3 - lr2);
            fu = calculate_training_error(u);
        }
        else if (((lr3 - u) * (u - ulim)) > 0.0)
        {
            fu = calculate_training_error(u);
            if (fu < etr3)
            {
                SHFT(lr2, lr3, u, lr3 + GOLD * (lr3 - lr2));
                SHFT(etr2, etr3, fu, calculate_training_error(u));
            }
        }
        else if (((u - ulim) * (ulim - lr3)) > 0.0)
        {
            u = ulim;
            fu = calculate_training_error(u);
        }
        else
        {
            u = lr3 + GOLD * (lr3 - lr2);
            fu = calculate_training_error(u);
        }
        SHFT(lr1, lr2, lr3, u);
        SHFT(etr1, etr2, etr3, fu);
    }
    return true;
}

/* Найти длину оптимального шага lr в заданном направлении m_aDirection[]
   по методу Брента (критерием оптимальности выступает ошибка обучения etr,
   которую надо минимизировать). В качестве стартовых точек метода Брента
   используются (lr1; etr1), (lr2; etr2) и (lr3; etr3). */
void CGradientDescentTraining::find_optimal_lr_by_brent(
        float lr1, float lr2, float lr3, float tol, float& lr, float& etr)
{
    float a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm;
    float e = 0.0;
    float eps = FLT_EPSILON;

    a = (lr1 < lr3 ? lr1 : lr3);
    b = (lr1 > lr3 ? lr1 : lr3);
    x = w = v = lr2;
    fw = fv = fx = calculate_training_error(x);
    for (int iter = 1; iter <= m_nMaxItersForLR; iter++)
    {
        xm = 0.5 * (a + b);
        tol2 = 2.0 * (tol1 = tol * fabs(x) * eps);
        if (fabs(x - xm) <= (tol2 - 0.5 * (b - a)))
        {
            lr = x;
            etr = fx;
            return;
        }
        if (fabs(e) > tol1)
        {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0)
            {
                p = -p;
            }
            q = fabs(q);
            etemp = e;
            e = d;
            if ((fabs(p) >= fabs(0.5 * q * etemp)) || (p <= (q * (a - x)))
                || (p >= (q * (b - x))))
            {
                d = CGOLD * (e = ((x >= xm) ? (a - x) : (b - x)));
            }
            else
            {
                d = p / q;
                u = x + d;
                if (((u - a) < tol2) || ((b - u) < tol2))
                {
                    d = SIGN(tol1, xm - x);
                }
            }
        }
        else
        {
            d = CGOLD * (e = ((x >= xm) ? (a - x) : (b - x)));
        }
        u = ((fabs(d) >= tol1) ? (x + d) : (x + SIGN(tol1, d)));
        fu = calculate_training_error(u);
        if (fu <= fx)
        {
            if (u >= x)
            {
                a = x;
            }
            else
            {
                b = x;
            }
            SHFT(v, w, x, u);
            SHFT(fv, fw, fx, fu);
        }
        else
        {
            if (u < x)
            {
                a = u;
            }
            else
            {
                b = u;
            }
            if ((fu <= fw) || (w == x))
            {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            }
            else if ((fu <= fv) || (v == x) || (v == w))
            {
                v = u;
                fv = fu;
            }
        }
    }
    lr = x;
    etr = fx;
}

/* Найти оптимальный шаг в заданном направлении m_aDirection по методу Брента.
Найденное значение оптимального шага записывается в передаваемый по ссылке
аргумент lr, а соответствующее значение целевой функции (функции ошибки) - в
передаваемый по ссылке аргумент etr. */
void CGradientDescentTraining::find_optimal_learning_rate(float& lr,float& etr)
{
    float lr1, lr2, lr3, etr1, etr2, etr3;
    find_init_lrs(lr1, lr2, lr3, etr1, etr2, etr3);
    if ((etr1 > getMeanError()) && (etr2 > getMeanError())
        && (etr3 > getMeanError()))
    {
        lr = 0.0;
        etr = getMeanError();
    }
    else
    {
        float tol = m_maxLearningRate / 100.0;
        if (tol > 0.1)
        {
            tol = 0.1;
        }
        else if (tol < FLT_EPSILON)
        {
            tol = FLT_EPSILON;
        }
        find_optimal_lr_by_brent(lr1, lr2, lr3, tol, lr, etr);
        if (lr <= 0.0)
        {
            lr = 0.0;
            etr = getMeanError();
        }
        else
        {
            if (etr > getMeanError())
            {
                lr = 0.0;
                etr = getMeanError();
            }
        }
    }
}

/* Вычислить среднеквадратичную ошибку обучения как функцию от длины шага
stepsize в направлении m_aDirection. */
float CGradientDescentTraining::calculate_training_error(float stepsize)
{
    int iStartWeight = 0, iWeight, i, j, k, n;
    for (i = 0; i < m_pTrainedMLP->getLayersCount(); i++)
    {
        n = m_pTrainedMLP->getInputsCountOfLayer(i);
        #pragma omp parallel for private(k,iWeight)
        for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
        {
            iWeight = iStartWeight + j * (n+1);
            for (k = 0; k <= n; k++)
            {
                m_pTempMLP->setWeight(
                        i, j, k, m_pTrainedMLP->getWeight(i,j,k)
                        + m_aDirection[iWeight++] * stepsize);
            }
        }
        iStartWeight += (m_pTrainedMLP->getLayerSize(i) * (n+1));
    }

    float mean_error = 0.0, cur_error, temp;
    i = m_pTempMLP->getLayersCount() - 1;
    for (int iSample = 0; iSample < getNumberOfTrainSamples(); iSample++)
    {
        calculate_outputs(iSample, m_pTempMLP);
        cur_error = 0.0;
        for (j = 0; j < m_pTempMLP->getLayerSize(i); j++)
        {
            temp = getTrainTarget(iSample,j) - getNetOutput(i,j);
            cur_error += (temp * temp);
        }
        cur_error /= m_pTempMLP->getLayerSize(i);
        mean_error += (cur_error * getSampleWeight(iSample));
    }
    mean_error /= getNumberOfTrainSamples();

    return mean_error;
}

/*****************************************************************************/
/* PUBLIC-ОПЕРАЦИИ КЛАССА CGradientDescentTraining */
/*****************************************************************************/

/*****************************************************************************/
/* Конструктор класса CGradientDescentTraining */
/*****************************************************************************/
CGradientDescentTraining::CGradientDescentTraining(QObject* pobj)
    :CBatchBackpropTraining(pobj)
{
    m_aDirection = 0;
    m_aOldG = 0;
    m_pTempMLP = 0;

    m_bConjugateGradient = true;
    m_maxLearningRate = 1.0;
    m_nMaxItersForLR = 20;
}

/*****************************************************************************/
/* Деструктор класса CGradientDescentTraining */
/*****************************************************************************/
CGradientDescentTraining::~CGradientDescentTraining()
{
    finalize_training();
}

/*****************************************************************************/
/* PROTECTED-ОПЕРАЦИИ КЛАССА CGradientDescentTraining */
/*****************************************************************************/

/*****************************************************************************/
// Выделить память для всех промежуточных переменных и инициализировать их
/*****************************************************************************/
void CGradientDescentTraining::initialize_training()
{
    CBatchBackpropTraining::initialize_training();

    int i, n = m_pTrainedMLP->getAllWeightsCount();
    m_aDirection = new float[n];
    for (i = 0; i < n; i++)
    {
        m_aDirection[i] = 0.0;
    }
    if (m_bConjugateGradient)
    {
        m_aOldG = new float[n];
    }

    m_pTempMLP = new CMultilayerPerceptron((*m_pTrainedMLP));
    (*m_pTempMLP) = (*m_pTrainedMLP);
}

/*****************************************************************************/
// Освободить память ото всех промежуточных переменных
/*****************************************************************************/
void CGradientDescentTraining::finalize_training()
{
    if (m_aDirection != 0)
    {
        delete[] m_aDirection;
        m_aDirection = 0;
    }
    if (m_aOldG != 0)
    {
        delete[] m_aOldG;
        m_aOldG = 0;
    }
    if (m_pTempMLP != 0)
    {
        delete m_pTempMLP;
        m_pTempMLP = 0;
    }

    CBatchBackpropTraining::finalize_training();
}

void CGradientDescentTraining::change_weights(int nEpoch,
                                              TTrainingState& training_state)
{
    int i, j, k, n, iWeight, iStartWeight;
    if (m_bConjugateGradient)
    {
        if (nEpoch > 1)
        {
            float betta, temp_val_1 = 0.0, temp_val_2 = 0.0;
            /* Вычисление коэффициента betta по формуле Полака-Рибъера.
            В temp_val_1 записывается числитель формулы, а в temp_val_2 -
            знаменатель.
               Заодно запоминаем в m_aOldG текущий вектор суммарного
            антиградиента (на следующем шаге m_aOldG будет использоваться как
            предыдущее значение антиградиента, что нужно для вычисления
            betta). */
            iStartWeight = 0;
            for (i = 0; i < m_pTrainedMLP->getLayersCount(); i++)
            {
                n = m_pTrainedMLP->getInputsCountOfLayer(i);
                #pragma omp parallel for \
                    private(k,iWeight) reduction(+:temp_val_1,temp_val_2)
                for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
                {
                    iWeight = iStartWeight + j * (n+1);
                    for (k=0; k <= n; k++)
                    {
                        temp_val_1+=getMeanGradient(i,j,k)
                                    *(getMeanGradient(i,j,k)-m_aOldG[iWeight]);
                        temp_val_2+=(m_aOldG[iWeight] * m_aOldG[iWeight]);
                        m_aOldG[iWeight] = getMeanGradient(i,j,k);
                        iWeight++;
                    }
                }
                iStartWeight += m_pTrainedMLP->getLayerSize(i) * (n+1);
            }

            /* Если вычисленное значение betta отрицательно, то во избежание
            зацикливания алгоритма делаем betta нулевым. */
            if (temp_val_2 != 0.0)
            {
                betta = temp_val_1 / temp_val_2;
                if (betta < 0.0)
                {
                    betta = 0.0;
                }
            }
            else
            {
                betta = 0.0;
            }

            /* Вычисляем новое направление в соответствии с методом сопряжённых
            градиентов. */
            iStartWeight = 0;
            for (i = 0; i < m_pTrainedMLP->getLayersCount(); i++)
            {
                n = m_pTrainedMLP->getInputsCountOfLayer(i);
                #pragma omp parallel for private(k,iWeight)
                for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
                {
                    iWeight = iStartWeight + j * (n+1);
                    for (k=0; k <= n; k++)
                    {
                        m_aDirection[iWeight] = getMeanGradient(i,j,k)
                                                + betta*m_aDirection[iWeight];
                        iWeight++;
                    }
                }
                iStartWeight += (m_pTrainedMLP->getLayerSize(i) * (n+1));
            }
        }
        else
        {
            /* Определяем начальное направление (в направлении, противоположном
               направлению суммарного градиента). Заодно запоминаем в m_aOldG
               начальный вектор суммарного антиградиента (на следующем шаге
               m_aOldG будет использоваться как предыдущее значение антиградиента,
               что нужно для вычисления betta). */
            iStartWeight = 0;
            for (i = 0; i < m_pTrainedMLP->getLayersCount(); i++)
            {
                n = m_pTrainedMLP->getInputsCountOfLayer(i);
                #pragma omp parallel for private(k,iWeight)
                for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
                {
                    iWeight = iStartWeight + j * (n+1);
                    for (k = 0; k <= n; k++)
                    {
                        m_aDirection[iWeight] = getMeanGradient(i,j,k);
                        m_aOldG[iWeight] = getMeanGradient(i,j,k);
                        iWeight++;
                    }
                }
                iStartWeight += (m_pTrainedMLP->getLayerSize(i) * (n+1));
            }
        }
    }
    else
    {
        /* Определяем направление, противоположное направлению суммарного
           градиента. Заодно запоминаем в m_aOldG. */
        iStartWeight = 0;
        for (i = 0; i < m_pTrainedMLP->getLayersCount(); i++)
        {
            n = m_pTrainedMLP->getInputsCountOfLayer(i);
            #pragma omp parallel for private(k,iWeight)
            for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
            {
                iWeight = iStartWeight + j * (n+1);
                for (k = 0; k <= n; k++)
                {
                    m_aDirection[iWeight] = getMeanGradient(i,j,k);
                    iWeight++;
                }
            }
            iStartWeight += (m_pTrainedMLP->getLayerSize(i) * (n+1));
        }
    }

    /* Определяем оптимальную длину шага в направлении m_aDirection. */
    float lr, new_error, new_weight;
    find_optimal_learning_rate(lr, new_error);
    if (new_error < getMeanError())
    {
        /* Обновляем веса нейронной сети */
        iStartWeight = 0;
        for (i = 0; i < m_pTrainedMLP->getLayersCount(); i++)
        {
            n = m_pTrainedMLP->getInputsCountOfLayer(i);
            #pragma omp parallel for private(k,iWeight,new_weight)
            for (j = 0; j < m_pTrainedMLP->getLayerSize(i); j++)
            {
                iWeight = iStartWeight + j * (n+1);
                for(k = 0; k <= m_pTrainedMLP->getInputsCountOfLayer(i); k++)
                {
                    new_weight = m_pTrainedMLP->getWeight(i,j,k)
                                 + lr * m_aDirection[iWeight];
                    m_pTrainedMLP->setWeight(i,j,k, new_weight);
                    iWeight++;
                }
            }
            iStartWeight += (m_pTrainedMLP->getLayerSize(i) * (n+1));
        }
    }
    else
    {
        training_state = tsMINIMUM;
    }
}

/*****************************************************************************/
/*                       ВНЕШНИЕ ФУНКЦИИ МОДУЛЯ annlib                       */
/*****************************************************************************/

/* Выполнить SOFTMAX-нормализацию вектора данных data[] длиной n. */
void do_softmax_normalization(float data[], int n)
{
    register int i;
    float sum = 0.0;
    for (i = 0; i < n; i++)
    {
        data[i] = exp(data[i]);
        sum += data[i];
    }
    for (i = 0; i < n; i++)
    {
        data[i] /= sum;
    }
}

/* Перемешивание элементов в некотором массиве длиной nLength путём генерации
случайных индексов этих элементов aIndexes. */
void calculate_rand_indexes(int aIndexes[], int nLength)
{
    int i, j, temp;

    for (i = 0; i < nLength; i++)
    {
        aIndexes[i] = i;
    }

    for (i = (nLength - 1); i > 0; i--)
    {
        j = get_random_value(0, i);
        temp = aIndexes[i];
        aIndexes[i] = aIndexes[j];
        aIndexes[j] = temp;
    }
}

/*****************************************************************************/
/* Вычислить среднеабсолютную ошибку регрессии */
/*****************************************************************************/
float regression_error(float output, float target)
{
    float result = fabs(output - target);
    if (result > FLT_EPSILON)
    {
        float temp = fabs(target);
        if (temp > FLT_EPSILON)
        {
            result /= fabs(target);
            if (ISNAN(result) || !(ISFINITE(result)))
            {
                result = 1000.0;
            }
            else
            {
                if (result > 1000.0)
                {
                    result = 1000.0;
                }
            }
        }
        else
        {
            result = 1000.0;
        }
    }
    else
    {
        result = 0.0;
    }
    return (100.0 * result);
}

/* Загрузить из файла sFileName обучающее множество - набор входных сигналов
aTrainInputs[] и соответствующих им желаемых выходных сигналов aTrainTargets[],
общее количество пар "входной сигнал - желаемый выходной сигнал" nTrainSamples,
размеры входного сигнала nTrainInputs и желаемого выходного сигнала
nTrainTargets.
   Для набора входных сигналов aTrainInputs[] и соответствующего ему набора
желаемых выходных сигналов aTrainTargets[] должна быть уже выделена область
памяти. Если же такая область памяти не выделена, т.е. aTrainInputs[] и
aTrainTargets[] - нулевые указатели, то функция просто считывает из файла
информацию о размерах обучающего множества и записывает её в nTrainSamples,
nTrainInputs и nTrainTargets.
   Если функция успешно выполнила свою работу, то возвращается true. В случае
ошибки (например, файл sFileName не существует, или в файле не те данные)
возвращается false. */
bool load_trainset(const QString& sFileName, float aTrainInputs[],
                   float aTrainTargets[], int& nTrainSamples,
                   int& nTrainInputs, int& nTrainTargets)
{
    QFile trainsetFile(sFileName);
    if (!trainsetFile.open(QFile::ReadOnly))
    {
        return false;
    }
    QDataStream trainsetStream(&trainsetFile);
    trainsetStream.setFloatingPointPrecision(QDataStream::SinglePrecision);
    if (trainsetStream.status() != QDataStream::Ok)
    {
        return false;
    }

    qint32 iTrainSamples, iTrainInputs, iTrainTargets;
    trainsetStream >> iTrainSamples >> iTrainInputs >> iTrainTargets;
    if (trainsetStream.status() != QDataStream::Ok)
    {
        return false;
    }
    if ((iTrainSamples <= 0) || (iTrainInputs <= 0) || (iTrainTargets < 0))
    {
        return false;
    }

    nTrainSamples = iTrainSamples;
    nTrainInputs = iTrainInputs;
    nTrainTargets = iTrainTargets;

    if (aTrainInputs == 0)
    {
        return true;
    }

    if (((aTrainTargets == 0) && (nTrainTargets > 0))
        || ((aTrainTargets != 0) && (nTrainTargets == 0)))
    {
        return false;
    }

    int i, j;
    float temp_value;
    bool result = true;
    for (i = 0; i < nTrainSamples; i++)
    {
        for (j = 0; j < nTrainInputs; j++)
        {
            trainsetStream >> temp_value;
            aTrainInputs[i * nTrainInputs + j] = temp_value;
        }
        if (trainsetStream.status() != QDataStream::Ok)
        {
            result = false;
        }
        else if (nTrainTargets > 0)
        {
            for (j = 0; j < nTrainTargets; j++)
            {
                trainsetStream >> temp_value;
                aTrainTargets[i * nTrainTargets + j] = temp_value;
            }
            if (trainsetStream.status() != QDataStream::Ok)
            {
                result = false;
            }
        }
        if (!result)
        {
            break;
        }
    }
    return result;
}

/* Сохранить в файл sFileName заданное обучающее множество. Размеры
сохраняемого обучающего множества - количество пар "входной сигнал - желаемый
выходной сигнал", количество элементов во входном сигнале и желаемом выходном
сигнале - передаётся аргументами nTrainSamples, nTrainInputs и nTrainTargets
соответственно. Набор входных сигналов и соответствующий ему набор желаемых
выходных сигналов содержатся в массивах, на которые указывают указатели
aTrainInputs[] и aTrainTargets[] соответственно.
   Если функция успешно выполнила свою работу, то возвращается true. В случае
ошибки (например, файл sFileName не существует, или размеры сохраняемого
обучающего множества невозможны), возвращается false */
bool save_trainset(const QString& sFileName, float aTrainInputs[],
                   float aTrainTargets[], int nTrainSamples,
                   int nTrainInputs, int nTrainTargets)
{
    if ((nTrainSamples <= 0) || (nTrainInputs <= 0) || (nTrainTargets < 0))
    {
        return false;
    }
    if ((aTrainInputs == 0) || ((aTrainTargets == 0) && (nTrainTargets > 0)))
    {
        return false;
    }

    QFile trainsetFile(sFileName);
    if (!trainsetFile.open(QFile::WriteOnly | QFile::Truncate))
    {
        return false;
    }
    QDataStream trainsetStream(&trainsetFile);
    trainsetStream.setFloatingPointPrecision(QDataStream::SinglePrecision);
    if (trainsetStream.status() != QDataStream::Ok)
    {
        return false;
    }

    qint32 iTrainSamples = nTrainSamples;
    qint32 iTrainInputs = nTrainInputs;
    qint32 iTrainTargets = nTrainTargets;
    trainsetStream << iTrainSamples << iTrainInputs << iTrainTargets;
    if (trainsetStream.status() != QDataStream::Ok)
    {
        return false;
    }

    int i, j;
    float temp_value;
    bool result = true;
    for (i = 0; i < nTrainSamples; i++)
    {
        for (j = 0; j < nTrainInputs; j++)
        {
            temp_value = aTrainInputs[i * nTrainInputs + j];
            trainsetStream << temp_value;
        }
        if (trainsetStream.status() != QDataStream::Ok)
        {
            result = false;
        }
        else if (nTrainTargets > 0)
        {
            for (j = 0; j < nTrainTargets; j++)
            {
                temp_value = aTrainTargets[i * nTrainTargets + j];
                trainsetStream << temp_value;
            }
            if (trainsetStream.status() != QDataStream::Ok)
            {
                result = false;
            }
        }
        if (!result)
        {
            break;
        }
    }
    return result;
}
