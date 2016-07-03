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

#include <cstring>
#include <ctime>
#include <ctype.h>
#include <fstream>
#include <iostream>

#include <QFile>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QTextStream>
#include <QDataStream>

#include "additional_unit.h"
#include "error_messages.h"
#include "commands_unit.h"
#include "mathlib_bond005.h"

using namespace std;

static const char *g_szAllInputsProcessingDuration = "Общее время обработки "\
        "всех входных сигналов составляет %1 сек.";
static const char *g_szOneInputProcessingDurationAtSecs = "Среднее время "\
        "обработки одного входного сигнала составляет %1 сек.";
static const char *g_szOneInputProcessingDurationAtMsecs = "Среднее время "\
        "обработки одного входного сигнала составляет %1 мсек.";
static const char *g_szTotalWeightsNumber = "Общее количество весов: %1.";
static const char *g_szLayersNumber = "Количество слоёв: %1.";
static const char *g_szInputsNumber = "Размер входного сигнала: %1.";
static const char *g_szLayerNumber = "%1-й слой";
static const char *g_szTrainSamplesNumber = "Количество обучающих примеров:"\
        "     %1.";
static const char *g_szTrainInputsNumber  = "Размер входного сигнала:      "\
        "     %1.";
static const char *g_szTrainTargetsNumber = "Размер желаемого выходного "\
        "сигнала: %1.";
static const char *g_szClassificationError = "Ошибка классификации составила"\
        " %1%.";
static const char *g_szRegressionError = "Ошибка регрессии составила %1%%.";
static const char *g_szMeanSquareError = "Среднеквадратичная ошибка "\
        "составила %1.";
static const char *g_szLayerName = "Слой %1:";
static const char *g_szNeuronName = "    Нейрон %1:";
static const char *g_szWeightName = "        Вес %1:";
static const char *g_szBiasName = "        Смещение:";

static const char *g_szClassificationTask = "class";
static const char *g_szRegressionTask = "reg";

static const char *g_szShowDivergentSamples = "show";
static const char *g_szRemoveDivergentSamples = "remove";
static const char *g_szUniteDivergentSamples = "unite";
static const char *g_szNoDivergentSamples = "В обучающем множестве нет "\
        "противоречивых примеров.";
static const char *g_szAllSamplesAreDivergent = "В обучающем множестве все "\
        "примеры являются противоречивыми.";
static const char *g_szGroupOfDivergentSamples1 = "%1-я группа (%2 пример):";
static const char *g_szGroupOfDivergentSamples2 = "%1-я группа (%2 примера):";
static const char *g_szGroupOfDivergentSamples3 = "%1-я группа (%2 примеров):";

/*****************************************************************************/
/* ВНУТРЕННИЕ ФУНКЦИИ МОДУЛЯ commands_unit.cpp */
/*****************************************************************************/

QString layer_description_to_string(const CMultilayerPerceptron& mlp,
                                    int iLayer)
{
    const char *szNeuron_1 = "нейрон";
    const char *szNeuron_2_4 = "нейрона";
    const char *szNeurons = "нейронов";
    const char *szSoftmax = " с функциями активации SOFTMAX.";
    const char *szSigmoid = " с сигмоидальными функциями активации.";
    const char *szLinear = " с линейными функциями активации.";
    const int nNeuronsNumber = mlp.getLayerSize(iLayer);
    QString sResult, sNeuronsNumber;
    int n;

    sNeuronsNumber = QString::number(nNeuronsNumber);
    n = sNeuronsNumber.size();
    if ((sNeuronsNumber.at(n-1) >= '1') && (sNeuronsNumber.at(n-1) <= '4'))
    {
        if (n >= 2)
        {
            if (sNeuronsNumber[n-2] == '1')
            {
                sResult = sNeuronsNumber + QString(" ") + QString(szNeurons);
            }
            else
            {
                if (sNeuronsNumber[n-1] == '1')
                {
                    sResult = sNeuronsNumber + QString(" ")
                            + QString(szNeuron_1);
                }
                else
                {
                    sResult = sNeuronsNumber + QString(" ")
                            + QString(szNeuron_2_4);
                }
            }
        }
        else
        {
            if (sNeuronsNumber[n-1] == '1')
            {
                sResult = sNeuronsNumber + QString(" ")
                        + QString(szNeuron_1);
            }
            else
            {
                sResult = sNeuronsNumber + QString(" ")
                        + QString(szNeuron_2_4);
            }
        }
    }
    else
    {
        sResult = sNeuronsNumber + QString(" ") + QString(szNeurons);
    }

    if (mlp.getActivationKind(iLayer) == SIG)
    {
        sResult += QString(szSigmoid);
    }
    else
    {
        if (mlp.getActivationKind(iLayer) == SOFT)
        {
            sResult += QString(szSoftmax);
        }
        else
        {
            sResult += QString(szLinear);
        }
    }

    return sResult;
}

/* Преобразование строки sStr в количество входов нейросети nInputsCount.
Исходная строка начинается с буквы "i", а затем следуют цифры, образующие
положительное целое число.
   В случае успешного преобразования возвращается true, а в случае ошибки -
false. */
bool string_to_inputs(const QString& sStr, int& nInputsCount)
{
    bool result = true;

    if (sStr.size() > 1)
    {
        if ((sStr[0] == 'i') || (sStr[0] == 'I'))
        {
            nInputsCount = sStr.mid(1, sStr.size() - 1).toInt(&result);
            if (result)
            {
                if (nInputsCount <= 0)
                {
                    result = false;
                }
            }
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

    return result;
}

/* Преобразование строки sStr в горизонтальный размер карты Кохонена xSize.
Исходная строка начинается с буквы "x", а затем следуют цифры, образующие
положительное целое число.
   В случае успешного преобразования возвращается true, а в случае ошибки -
false. */
bool string_to_xsize(const QString& sStr, int& xSize)
{
    bool result = true;

    if (sStr.size() > 1)
    {
        if ((sStr[0] == 'x') || (sStr[0] == 'X'))
        {
            xSize = sStr.mid(1, sStr.size() - 1).toInt(&result);
            if (result)
            {
                if (xSize <= 0)
                {
                    result = false;
                }
            }
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

    return result;
}

/* Преобразование строки sStr в вертикальный размер карты Кохонена ySize.
Исходная строка начинается с буквы "x", а затем следуют цифры, образующие
положительное целое число.
   В случае успешного преобразования возвращается true, а в случае ошибки -
false. */
bool string_to_ysize(const QString& sStr, int& ySize)
{
    bool result = true;

    if (sStr.size() > 1)
    {
        if ((sStr[0] == 'y') || (sStr[0] == 'Y'))
        {
            ySize = sStr.mid(1, sStr.size() - 1).toInt(&result);
            if (result)
            {
                if (ySize <= 0)
                {
                    result = false;
                }
            }
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

    return result;
}

/* Преобразование строки sStr в структуру слоя нейросети: количество нейронов
nLayerSize и тип активационной функции нейронов activationKind.
   Исходная строка описания слоя начинается с цифр, образующие положительное
целое число (размер слоя), и завершается подстрокой "sig" (активационная
функция - сигмоида) либо подстрокой "lin" (активационная функция - линейна).
   В случае успешного преобразования возвращается true, а в случае ошибки -
false. */
bool string_to_layer(const QString& sStr, int& nLayerSize,
                     TActivationKind& activationKind)
{
    bool result = true;

    if (sStr.size() >= 4)
    {
        QString sActivation = sStr.right(3);
        nLayerSize = sStr.left(sStr.size() - 3).toInt(&result);
        if (result)
        {
            if (nLayerSize > 0)
            {
                if (sActivation.compare("lin", Qt::CaseInsensitive) == 0)
                {
                    result = true;
                    activationKind = LIN;
                }
                else
                {
                    if (sActivation.compare("sig", Qt::CaseInsensitive) == 0)
                    {
                        result = true;
                        activationKind = SIG;
                    }
                    else
                    {
                        if (sActivation.compare("max", Qt::CaseInsensitive) == 0)
                        {
                            result = true;
                            activationKind = SOFT;
                        }
                        else
                        {
                            result = false;
                        }
                    }
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

    return result;
}

/* Преобразование строки sStr в описание структуры нейронной сети: число входов
nInputsCount, число слоёв nLayersCount, размеры слоёв aSizesOfLayers[] и
активационные функции нейронов в слоях aActivations[].
   Пример исходной строки, описывающей многослойный персептрон с тремя входами,
четырмя нейронами в единственном скрытом слое и двумя - в выходном слое, причём
активационная функция нейронов скрытого слоя - сигмоида, а выходного слоя -
линейная:
   i3-4sig-2lin
   Если преобразование выполнилось успешно, то функция возвращает true. В
противном случае функция возвращает false. */
bool parse_mlp_structure_description(const QString& sStr, int& nInputsCount,
                                     int& nLayersCount, int aSizesOfLayers[],
                                     TActivationKind aActivations[])
{
    int result = true, n = sStr.size();
    if (n > 0)
    {
        QVector<QString> strParts;
        if ((sStr[0] != '-') && (sStr[n-1] != '-'))
        {
            int start_ind = -1;
            QString sCur;
            for (int i = 0; i < n; i++)
            {
                if (sStr[i] == '-')
                {
                    if (start_ind >= 0)
                    {
                        sCur = sStr.mid(start_ind, i - start_ind);
                        start_ind = -1;
                        strParts.push_back(sCur);
                    }
                    else
                    {
                        result = false;
                        break;
                    }
                }
                else
                {
                    if (start_ind < 0)
                    {
                        start_ind = i;
                    }
                }
            }
            if (start_ind >= 0)
            {
                sCur = sStr.mid(start_ind, n - start_ind + 1);
                strParts.push_back(sCur);
            }
            if (strParts.size() >= 2)
            {
                if (string_to_inputs(strParts[0], nInputsCount))
                {
                    result = true;
                    nLayersCount = strParts.size() - 1;
                    int nLayerSize;
                    TActivationKind activationKind;
                    for (int i = 0; i < nLayersCount; i++)
                    {
                        if (string_to_layer(strParts[i+1], nLayerSize,
                                            activationKind))
                        {
                            if ((activationKind == SOFT)
                                    && (i < (nLayersCount - 1)))
                            {
                                result = false;
                                break;
                            }
                            if (aSizesOfLayers != NULL)
                            {
                                aSizesOfLayers[i] = nLayerSize;
                            }
                            if (aActivations != NULL)
                            {
                                aActivations[i] = activationKind;
                            }
                        }
                        else
                        {
                            result = false;
                            break;
                        }
                    }
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
        else
        {
            result = false;
        }
    }
    return result;
}

/* Преобразование строки sStr в описание структуры карты Кохонена: число входов
nInputsCount, горизонтальный размер карты xSize и вертикальный размер карты
ySize.
   Пример исходной строки, описывающей карту Кохонена с тремя входами и
размером нейронной решётки 4 нейрона по горизонтали и 2 нейрона по вертикали:
   i3-x4-y2
   Если преобразование выполнилось успешно, то функция возвращает true. В
противном случае функция возвращает false. */
bool parse_som_structure_description(const QString& sStr, int& nInputsCount,
                                     int& xSize, int& ySize)
{
    int result = true, n = sStr.size();
    if (n > 0)
    {
        QVector<QString> strParts;
        if ((sStr[0] != '-') && (sStr[n-1] != '-'))
        {
            int start_ind = -1;
            QString sCur;
            for (int i = 0; i < n; i++)
            {
                if (sStr[i] == '-')
                {
                    if (start_ind >= 0)
                    {
                        sCur = sStr.mid(start_ind, i - start_ind);
                        start_ind = -1;
                        strParts.push_back(sCur);
                    }
                    else
                    {
                        result = false;
                        break;
                    }
                }
                else
                {
                    if (start_ind < 0)
                    {
                        start_ind = i;
                    }
                }
            }
            if (start_ind >= 0)
            {
                sCur = sStr.mid(start_ind, n - start_ind + 1);
                strParts.push_back(sCur);
            }
            if (strParts.size() == 3)
            {
                result = (string_to_inputs(strParts[0], nInputsCount)
                          && string_to_xsize(strParts[1], xSize)
                          && string_to_ysize(strParts[2], ySize));
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
    return result;
}

/* Преобразование строки sStr с описанием последовательности целых чисел,
разделённых знаком "-" (тире), в соответствующий массив целых чисел aIntArray.
   Если преобразование выполнилось успешно, то функция возвращает
неотрицательное целое число - длину массива aIntArray (ноль возвращается, если
строка пуста). Если же преобразование не смогло выполниться (исходная строка
содержит некорректное описание последовательности целых чисел или же вообще его
не содержит), то функция возвращает отрицательное число (как правило, -1). */
int str_to_int_array(const QString& sStr, int aIntArray[])
{
    bool ok = true;
    int result = 0, n = sStr.size();
    if (n > 0)
    {
        if ((sStr[0] != '-') && (sStr[n-1] != '-'))
        {
            int start_ind = -1;
            int temp_value;
            QString sCurNumber;
            for (int i = 0; i < n; i++)
            {
                if (sStr[i] == '-')
                {
                    if (start_ind >= 0)
                    {
                        sCurNumber = sStr.mid(start_ind, i - start_ind);
                        start_ind = -1;
                        temp_value = sCurNumber.toInt(&ok);
                        if (ok)
                        {
                            if (aIntArray != NULL)
                            {
                                aIntArray[result] = temp_value;
                            }
                            result++;
                        }
                        else
                        {
                            result = -1;
                            break;
                        }
                    }
                    else
                    {
                        result = -1;
                        break;
                    }
                }
                else
                {
                    if (start_ind < 0)
                    {
                        start_ind = i;
                    }
                }
            }
            if (start_ind >= 0)
            {
                sCurNumber = sStr.mid(start_ind, n - start_ind + 1);
                temp_value = sCurNumber.toInt(&ok);
                if (ok)
                {
                    if (aIntArray != NULL)
                    {
                        aIntArray[result] = temp_value;
                    }
                    result++;
                }
                else
                {
                    result = -1;
                }
            }
        }
        else
        {
            result = -1;
        }
    }
    return result;
}

/* Проверка корректности заданной структуры многослойного персептрона.
   ВХОДНЫЕ АРГУМЕНТЫ
   1. nLayersCount - количество слоёв многослойного персептрона.
   2. aStructure - массив целых чисел длиной (nLayersCount + 1); первый
элемент массива - количество входов, остальные - размеры слоёв нейросети
с первого по последний соответственно.
   ВОЗВРАЩАЕМЫЙ РЕЗУЛЬТАТ
   Если nLayersCount > 0 и aStructure[i] > 0 для всех 0 <= i <= nLayersCount,
то считается, что структура многослойного персептрона задана правильно,
и тогда функция возвращает true. В любом другом случае функция возвращает
false.
   ПРИМЕЧАНИЕ
   Если структура многослойного персептрона некорректна, то функция, кроме
того, что возвращает значение false, ещё и печатает сообщение об ошибке
в стандартный поток ошибок stderr. */
bool check_mlp_structure(int nLayersCount, int aStructure[])
{
    bool result;
    if (nLayersCount > 0)
    {
        if (aStructure[0] > 0)
        {
            result = true;
            for (int i = 1; i <= nLayersCount; i++)
            {
                if (aStructure[i] <= 0)
                {
                    result = false;
                    break;
                }
            }
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
    if (!result)
    {
        cerr << qPrintable(QString(g_szMlpStructureError));
    }
    return result;
}

/* Проверка корректности индексов элемента нейронной сети.
   ВХОДНЫЕ АРГУМЕНТЫ
   1. aIndexes - массив целых чисел, содержащий индексы элементов нейронной
сети в следующей последовательности: слой-нейрон-вход;
   2. nIndexesCount - количество индексов и, соответственно, длина массива
aIndexes:
   а) если nIndexesCount == 3, то индексы указываеют на aIndexes[2]-й вес
aIndexes[1]-го нейрона aIndexes[0]-го слоя;
   б) если nIndexesCount == 2, то индексы указывают на aIndexes[1]-й нейрон
aIndexes[0]-го слоя;
   в) если nIndexesCount == 1, то единственный индекс указывает на
aIndexes[0]-й слой;
   г) и, наконец, если nIndexesCount == 0, то считается, что указываются
веса всех нейронов многослойного персептрона.
   3. mlp - объект класса "многослойный персептрон", передаваемый по ссылке;
именно для него проверяется корректность индексов aIndexes.
   ВОЗВРАЩАЕМОЕ ЗНАЧЕНИЕ
   Функция возвращает true, если индексы элемента нейронной сети заданы
правильно, и false - в противном случае. Правильными индексами являются такие
индексы, для которых выполняются следующие условия:
   1) 0 <=nIndexesCount <= 3;
   2) 0 <= aIndexes[0] <  mlp.getLayersCount();
   3) 0 <= aIndexes[1] <  mlp.getLayerSize(aIndexes[0]);
   4) 0 <= aIndexes[2] <= mlp.getInputsCountOfLayer(aIndexes[0]) (поскольку
каждый нейрон всегда имеет дополнительный вход, равный +1, и его вес - это
смещение нейрона).
   ПРИМЕЧАНИЕ
   Если индексы элемента многослойного персептрона некорректны, то функция,
кроме того, что возвращает значение false, ещё и печатает сообщение об ошибке
в стандартный поток ошибок stderr. */
bool check_mlp_indexes(int aIndexes[], int nIndexesCount,
                       const CMultilayerPerceptron& rMlp)
{
    bool result;
    if ((nIndexesCount >= 0) && (nIndexesCount <= 3))
    {
        if (nIndexesCount > 0)
        {
            int i = aIndexes[0];
            if ((i >= 0) && (i < rMlp.getLayersCount()))
            {
                if (nIndexesCount > 1)
                {
                    int j = aIndexes[1];
                    if ((j >= 0) && (j < rMlp.getLayerSize(i)))
                    {
                        if (nIndexesCount > 2)
                        {
                            int k = aIndexes[2];
                            result = ((k >= 0)
                                      && (k <= rMlp.getInputsCountOfLayer(i)));
                            if (!result)
                            {
                                cerr<<qPrintable(QString(g_szInputIndexError));
                            }
                        }
                        else
                        {
                            result = true;
                        }
                    }
                    else
                    {
                        cerr << qPrintable(QString(g_szNeuronIndexError));
                        result = false;
                    }
                }
                else
                {
                    result = true;
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szLayerIndexError));
                result = false;
            }
        }
        else
        {
            result = true;
        }
    }
    else
    {
        cerr << qPrintable(QString(g_szIndexesError));
        result = false;
    }
    return result;
}

/* Проверка правильности структуры перечня "ключ-значение" rCmdParams для
функции StrStructureOfMLP (изменения структуры нейросети). */
bool check_params_for_setStructureOfMLP(const TCmdParams& rCmdParams)
{
    bool result;

    // количество аргументов (пар "ключ-значение") равно 3
    if (rCmdParams.size() == 3)
    {
        // есть ключ "mlp"
        if (rCmdParams.contains("mlp"))
        {
            // длина строки-значения ключа "mlp" ненулевая
            if (!rCmdParams["mlp"].isEmpty())
            {
                // есть ключ "set"
                if (rCmdParams.contains("set"))
                {
                    // длина строки-значения ключа "set" ненулевая
                    if (!rCmdParams["set"].isEmpty())
                    {
                        // есть ключ "struct"
                        if (rCmdParams.contains("struct"))
                        {
                            // ключ "struct" не имеет значения (пустая строка)
                            if (!rCmdParams["struct"].isEmpty())
                            {
                                result = false;
                                cerr << qPrintable(
                                            QString(g_szImpossibleVal).arg(
                                                "struct"));
                            }
                            else
                            {
                                result = true;
                            }
                        }
                        else
                        {
                            cerr << qPrintable(
                                        QString(g_szArgIsNotFound).arg(
                                            "struct"));
                            result = false;
                        }
                    }
                    else
                    {
                        result = false;
                        cerr << qPrintable(QString(g_szNullVal).arg("set"));
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szArgIsNotFound).arg("set"));
                    result = false;
                }
            }
            else
            {
                result = false;
                cerr << qPrintable(QString(g_szNullVal).arg("mlp"));
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("mlp"));
            result = false;
        }
    }
    else
    {
        result = false;
        if (rCmdParams.size() < 3)
        {
            cerr << qPrintable(QString(g_szFewArgs));
        }
        else
        {
            cerr << qPrintable(QString(g_szManyArgs));
        }
    }

    return result;
}

/* Проверка правильности структуры перечня "ключ-значение" rCmdParams для
функции getStructureOfMLP (получение информации о структуре нейросети). */
bool check_params_for_getStructureOfMLP(const TCmdParams& rCmdParams)
{
    bool result;

    // количество аргументов (пар "ключ-значение") равно 3
    if (rCmdParams.size() == 3)
    {
        // есть ключ "mlp"
        if (rCmdParams.contains("mlp"))
        {
            // длина строки-значения ключа "mlp" ненулевая
            if (!rCmdParams["mlp"].isEmpty())
            {
                // есть ключ "get"
                if (rCmdParams.contains("get"))
                {
                    // ключ "get" не имеет значения (пустая строка)
                    if (!rCmdParams["get"].isEmpty())
                    {
                        result = false;
                        cerr << qPrintable(QString(g_szImpossibleVal).arg(
                                               "get"));
                    }
                    else
                    {
                        // есть ключ "struct"
                        if (rCmdParams.contains("struct"))
                        {
                            // ключ "struct" не имеет значения (пустая строка)
                            if (!rCmdParams["struct"].isEmpty())
                            {
                                result = false;
                                cerr << qPrintable(
                                            QString(g_szImpossibleVal).arg(
                                                "struct"));
                            }
                            else
                            {
                                result = true;
                            }
                        }
                        else
                        {
                            cerr << qPrintable(
                                        QString(g_szArgIsNotFound).arg(
                                            "struct"));
                            result = false;
                        }
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szArgIsNotFound).arg("get"));
                    result = false;
                }
            }
            else
            {
                result = false;
                cerr << qPrintable(QString(g_szNullVal).arg("mlp"));
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("mlp"));
            result = false;
        }
    }
    else
    {
        result = false;
        if (rCmdParams.size() < 3)
        {
            cerr << qPrintable(QString(g_szFewArgs));

        }
        else
        {
            cerr << qPrintable(QString(g_szManyArgs));
        }
    }

    return result;
}

/* Проверка правильности структуры перечня "ключ-значение" rCmdParams для
функции setWeightOfMLP (изменения весов нейросети). */
bool check_params_for_setWeightOfMLP(const TCmdParams& rCmdParams)
{
    bool result;

    // количество аргументов (пар "ключ-значение") равно 3
    if (rCmdParams.size() == 3)
    {
        // есть ключ "mlp"
        if (rCmdParams.contains("mlp"))
        {
            // длина строки-значения ключа "mlp" ненулевая
            if (!rCmdParams["mlp"].isEmpty())
            {
                // есть ключ "set"
                if (rCmdParams.contains("set"))
                {
                    // длина строки-значения ключа "set" ненулевая
                    if (!rCmdParams["set"].isEmpty())
                    {
                        // есть ключ "w"
                        if (rCmdParams.contains("w"))
                        {
                            result = true;
                        }
                        else
                        {
                            cerr << qPrintable(
                                        QString(g_szArgIsNotFound).arg("w"));
                            result = false;
                        }
                    }
                    else
                    {
                        result = false;
                        cerr << qPrintable(QString(g_szNullVal).arg("set"));
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szArgIsNotFound).arg("set"));
                    result = false;
                }
            }
            else
            {
                result = false;
                cerr << qPrintable(QString(g_szNullVal).arg("mlp"));
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("mlp"));
            result = false;
        }
    }
    else
    {
        result = false;
        if (rCmdParams.size() < 3)
        {
            cerr << qPrintable(QString(g_szFewArgs));
        }
        else
        {
            cerr << qPrintable(QString(g_szManyArgs));
        }
    }

    return result;
}

/* Проверка правильности структуры перечня "ключ-значение" rCmdParams для
функции getWeightOfMLP (получение информации о весах нейросети). */
bool check_params_for_getWeightOfMLP(const TCmdParams& rCmdParams)
{
    bool result;

    // количество аргументов (пар "ключ-значение") равно 3
    if (rCmdParams.size() == 3)
    {
        // есть ключ "mlp"
        if (rCmdParams.contains("mlp"))
        {
            // длина строки-значения ключа "mlp" ненулевая
            if (!rCmdParams["mlp"].isEmpty())
            {
                // есть ключ "get"
                if (rCmdParams.contains("get"))
                {
                    // ключ "get" не имеет значения (пустая строка)
                    if (!rCmdParams["get"].isEmpty())
                    {
                        result = false;
                        cerr << qPrintable(
                                    QString(g_szImpossibleVal).arg("get"));
                    }
                    else
                    {
                        // есть ключ "w"
                        if (rCmdParams.contains("w"))
                        {
                            result = true;
                        }
                        else
                        {
                            cerr << qPrintable(
                                        QString(g_szArgIsNotFound).arg("w"));
                            result = false;
                        }
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szArgIsNotFound).arg("get"));
                    result = false;
                }
            }
            else
            {
                result = false;
                cerr << qPrintable(QString(g_szNullVal).arg("mlp"));
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("mlp"));
            result = false;
        }
    }
    else
    {
        result = false;
        if (rCmdParams.size() < 3)
        {
            cerr << qPrintable(QString(g_szFewArgs));
        }
        else
        {
            cerr << qPrintable(QString(g_szManyArgs));
        }
    }

    return result;
}

/* Проверка правильности структуры перечня "ключ-значение" rCmdParams для
функции initialize_mlp (инициализация весов нейросети). */
bool check_params_for_InitializeMLP(const TCmdParams& rCmdParams)
{
    bool result;

    // количество аргументов (пар "ключ-значение") равно 2
    if (rCmdParams.size() == 2)
    {
        // есть ключ "mlp"
        if (rCmdParams.contains("mlp"))
        {
            // длина строки-значения ключа "mlp" ненулевая
            if (!rCmdParams["mlp"].isEmpty())
            {
                if (rCmdParams.contains("init"))
                {
                    // ключ "init" не имеет значения (пустая строка)
                    if (!rCmdParams["init"].isEmpty())
                    {
                        result = false;
                        cerr << qPrintable(QString(g_szImpossibleVal).arg(
                                               "init"));
                    }
                    else
                    {
                        result = true;
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szArgIsNotFound).arg("init"));
                    result = false;
                }
            }
            else
            {
                result = false;
                cerr << qPrintable(QString(g_szNullVal).arg("mlp"));
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("mlp"));
            result = false;
        }
    }
    else
    {
        result = false;
        if (rCmdParams.size() < 2)
        {
            cerr << qPrintable(QString(g_szFewArgs));
        }
        else
        {
            cerr << qPrintable(QString(g_szManyArgs));
        }
    }

    return result;
}

/* Проверка правильности структуры перечня "ключ-значение" rCmdParams для
функции use_mlp (использование нейросети). */
bool check_params_for_UseMLP(const TCmdParams& rCmdParams)
{
    bool result = true, bCalculateOutputs = true;
    int nArgsCount = rCmdParams.size();

    // есть ключ "mlp"
    if (rCmdParams.contains("mlp"))
    {
        nArgsCount--;
        // длина строки-значения ключа "mlp" ненулевая
        if (!rCmdParams["mlp"].isEmpty())
        {
            result = true;
        }
        else
        {
            result = false;
            cerr << qPrintable(QString(g_szNullVal).arg("mlp"));
        }
    }
    else
    {
        cerr << qPrintable(QString(g_szArgIsNotFound).arg("mlp"));
        result = false;
    }

    if (result)
    {
        // есть ключ "in"
        if (rCmdParams.contains("in"))
        {
            nArgsCount--;
            // длина строки-значения ключа "in" ненулевая
            if (rCmdParams["in"].isEmpty())
            {
                result = false;
                cerr << qPrintable(QString(g_szNullVal).arg("in"));
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("in"));
            result = false;
        }
    }

    if (result)
    {
        // есть ключ "out"
        if (rCmdParams.contains("out"))
        {
            nArgsCount--;
            // длина строки-значения ключа "out" ненулевая
            if (rCmdParams["out"].isEmpty())
            {
                result = false;
                cerr << qPrintable(QString(g_szNullVal).arg("out"));
            }
            else
            {
                bCalculateOutputs = true;
            }
        }
        else
        {
            bCalculateOutputs = false;
        }
    }

    if (result)
    {
        // есть ключ "task"
        if (rCmdParams.contains("task"))
        {
            nArgsCount--;
            // длина строки-значения ключа "task" ненулевая
            if (rCmdParams["task"].isEmpty())
            {
                result = false;
                cerr << qPrintable(QString(g_szNullVal).arg("task"));
            }
        }
    }

    if (result)
    {
        if (nArgsCount > 0)
        {
            cerr << qPrintable(QString(g_szManyArgs));
            result = false;
        }
    }

    return result;
}

/* Проверка правильности структуры перечня "ключ-значение" rCmdParams для
функции showTrainset (вывод информации о структуре обучающего множества). */
bool check_params_for_ShowTrainset(const TCmdParams& rCmdParams)
{
    bool result;

    // количество аргументов (пар "ключ-значение") равно 1 или 2
    if (rCmdParams.size() == 1)
    {
        // есть ключ "trainset"
        if (rCmdParams.contains("trainset"))
        {
            // длина строки-значения ключа "trainset" ненулевая
            if (!rCmdParams["trainset"].isEmpty())
            {
                result = true;
            }
            else
            {
                result = false;
                cerr << qPrintable(QString(g_szNullVal).arg("trainset"));
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("trainset"));
            result = false;
        }
    }
    else
    {
        result = false;
        if (rCmdParams.size() < 1)
        {
            cerr << qPrintable(QString(g_szFewArgs));
        }
        else
        {
            cerr << qPrintable(QString(g_szManyArgs));
        }
    }

    return result;
}

/* Проверка правильности структуры перечня "ключ-значение" rCmdParams для
функции separate_trainset, выполняющей разделение обучающего множества на
собственно обучающее и тестовое (контрольное) подмножества. */
bool check_params_for_SeparateTrainset(const TCmdParams& rCmdParams)
{
    bool result;

    // количество аргументов (пар "ключ-значение") равно 3
    if (rCmdParams.size() == 3)
    {
        // есть ключ "trainset"
        if (rCmdParams.contains("trainset"))
        {
            // длина строки-значения ключа "trainset" ненулевая
            if (!rCmdParams["trainset"].isEmpty())
            {
                result = true;
            }
            else
            {
                result = false;
                cerr << qPrintable(QString(g_szNullVal).arg("trainset"));
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("trainset"));
            result = false;
        }

        if (result)
        {
            // есть ключ "controlset"
            if (rCmdParams.contains("controlset"))
            {
                // длина строки-значения ключа "controlset" ненулевая
                if (!rCmdParams["controlset"].isEmpty())
                {
                    result = true;
                }
                else
                {
                    result = false;
                    cerr << qPrintable(QString(g_szNullVal).arg("controlset"));
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szArgIsNotFound).arg(
                                       "controlset"));
                result = false;
            }
        }

        if (result)
        {
            // есть ключ "r"
            if (rCmdParams.contains("r"))
            {
                // длина строки-значения ключа "r" ненулевая
                if (rCmdParams["r"].isEmpty())
                {
                    result = false;
                    cerr << qPrintable(QString(g_szNullVal).arg("r"));
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szArgIsNotFound).arg("r"));
                result = false;
            }
        }
    }
    else
    {
        result = false;
        if (rCmdParams.size() < 3)
        {
            cerr << qPrintable(QString(g_szFewArgs));
        }
        else
        {
            cerr << qPrintable(QString(g_szManyArgs));
        }
    }

    return result;
}

/* Проверка правильности структуры перечня "ключ-значение" rCmdParams для
функции CSVtoTrainset, выполняющей преобразование данных из формата CSV
в формат обучающего множества. */
bool check_params_for_CSVtoTrainset(const TCmdParams& rCmdParams)
{
    bool result;
    if (rCmdParams.size() != 5)
    {
        if (rCmdParams.size() < 5)
        {
            cerr << qPrintable(QString(g_szFewArgs));
        }
        else
        {
            cerr << qPrintable(QString(g_szManyArgs));
        }
        result = false;
    }
    else
    {
        if (rCmdParams.contains("csv"))
        {
            if (rCmdParams["csv"].isEmpty())
            {
                cerr << qPrintable(QString(g_szNullVal).arg("csv"));
                result = false;
            }
            else
            {
                result = true;
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("csv"));
            result = false;
        }
        if (result)
        {
            if (rCmdParams.contains("trainset"))
            {
                if (rCmdParams["trainset"].isEmpty())
                {
                    cerr << qPrintable(QString(g_szNullVal).arg("trainset"));
                    result = false;
                }
                else
                {
                    result = true;
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szArgIsNotFound).arg("trainset"));
                result = false;
            }
        }
        if (result)
        {
            if (rCmdParams.contains("i"))
            {
                if (rCmdParams["i"].isEmpty())
                {
                    cerr << qPrintable(QString(g_szNullVal).arg("i"));
                    result = false;
                }
                else
                {
                    result = true;
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szArgIsNotFound).arg("i"));
                result = false;
            }
        }
        if (result)
        {
            if (rCmdParams.contains("o"))
            {
                if (rCmdParams["o"].isEmpty())
                {
                    cerr << qPrintable(QString(g_szNullVal).arg("o"));
                    result = false;
                }
                else
                {
                    result = true;
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szArgIsNotFound).arg("o"));
                result = false;
            }
        }
        if (result)
        {
            if (rCmdParams.contains("to_ts"))
            {
                if (!rCmdParams["to_ts"].isEmpty())
                {
                    cerr<<qPrintable(QString(g_szImpossibleVal).arg("to_ts"));
                    result = false;
                }
                else
                {
                    result = true;
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szArgIsNotFound).arg("to_ts"));
                result = false;
            }
        }
    }
    return result;
}

/* Проверка правильности структуры перечня "ключ-значение" rCmdParams для
функции trainsetToCSV, выполняющей преобразование данных из формата обучающего
множества в формат CSV. */
bool check_params_for_trainsetToCSV(const TCmdParams& rCmdParams)
{
    bool result;
    if (rCmdParams.size() != 3)
    {
        if (rCmdParams.size() < 3)
        {
            cerr << qPrintable(QString(g_szFewArgs));
        }
        else
        {
            cerr << qPrintable(QString(g_szManyArgs));
        }
        result = false;
    }
    else
    {
        if (rCmdParams.contains("csv"))
        {
            if (rCmdParams["csv"].isEmpty())
            {
                cerr << qPrintable(QString(g_szNullVal).arg("csv"));
                result = false;
            }
            else
            {
                result = true;
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("csv"));
            result = false;
        }
        if (result)
        {
            if (rCmdParams.contains("trainset"))
            {
                if (rCmdParams["trainset"].isEmpty())
                {
                    cerr << qPrintable(QString(g_szNullVal).arg("trainset"));
                    result = false;
                }
                else
                {
                    result = true;
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szArgIsNotFound).arg("trainset"));
                result = false;
            }
        }
        if (result)
        {
            if (rCmdParams.contains("to_csv"))
            {
                if (!rCmdParams["to_csv"].isEmpty())
                {
                    cerr<<qPrintable(QString(g_szImpossibleVal).arg("to_csv"));
                    result = false;
                }
                else
                {
                    result = true;
                }
            }
            else
            {
                result = false;
            }
        }
    }
    return result;
}

/* Проверка правильности структуры перечня "ключ-значение" rCmdParams для
функции processDivergentTrainSamples, выполняющей выявление и обработку
противоречивых примеров в обучающем множестве. */
bool check_params_for_processDivergentTrainSamples(const TCmdParams&rCmdParams)
{
    bool result;
    int nNumberOfCmdParams = rCmdParams.size();
    if ((nNumberOfCmdParams != 2) && (nNumberOfCmdParams != 3))
    {
        if (nNumberOfCmdParams < 2)
        {
            cerr << qPrintable(QString(g_szFewArgs));
        }
        else
        {
            cerr << qPrintable(QString(g_szManyArgs));
        }
        result = false;
    }
    else
    {
        if (rCmdParams.contains("divergent"))
        {
            if (rCmdParams["divergent"].isEmpty())
            {
                cerr << qPrintable(QString(g_szNullVal).arg("divergent"));
                result = false;
            }
            else
            {
                QString sValue = rCmdParams["divergent"];
                if ((sValue.compare(g_szShowDivergentSamples,
                                    Qt::CaseInsensitive) == 0)
                        || (sValue.compare(g_szRemoveDivergentSamples,
                                           Qt::CaseInsensitive) == 0)
                        || (sValue.compare(g_szUniteDivergentSamples,
                                           Qt::CaseInsensitive) == 0))
                {
                    nNumberOfCmdParams--;
                    if ((sValue.compare(g_szUniteDivergentSamples,
                                        Qt::CaseInsensitive) == 0))
                    {
                        if (nNumberOfCmdParams > 0)
                        {
                            if (rCmdParams.contains("task"))
                            {
                                nNumberOfCmdParams--;
                                if ((rCmdParams["task"].compare(
                                         g_szClassificationTask,
                                         Qt::CaseInsensitive) == 0)
                                        || (rCmdParams["task"].compare(
                                                g_szRegressionTask,
                                                Qt::CaseInsensitive) == 0))
                                {
                                    result = true;
                                }
                                else
                                {
                                    result = false;
                                    cerr << qPrintable(g_szIncorrectTask);
                                }
                            }
                            else
                            {
                                cerr << qPrintable(
                                            QString(g_szArgIsNotFound).arg(
                                                "task"));
                                result = false;
                            }
                        }
                        else
                        {
                            result = false;
                            cerr << qPrintable(QString(g_szFewArgs));
                        }
                    }
                    else
                    {
                        result = true;
                    }
                }
                else
                {
                    result = false;
                    cerr<<qPrintable(QString(g_szUnknownDivergentProcessing));
                }
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("divergent"));
            result = false;
        }
        if (result)
        {
            if (rCmdParams.contains("trainset"))
            {
                if (rCmdParams["trainset"].isEmpty())
                {
                    cerr << qPrintable(QString(g_szNullVal).arg("trainset"));
                    result = false;
                }
                else
                {
                    nNumberOfCmdParams--;
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szArgIsNotFound).arg("trainset"));
                result = false;
            }
        }
        if (result && (nNumberOfCmdParams > 0))
        {
            result = false;
            cerr << qPrintable(QString(g_szManyArgs));
        }
    }
    return result;
}

bool check_params_for_deleteRepeatingTrainSamples(const TCmdParams&rCmdParams)
{
    bool result;
    int nNumberOfCmdParams = rCmdParams.size();
    if (nNumberOfCmdParams != 2)
    {
        if (nNumberOfCmdParams < 2)
        {
            cerr << qPrintable(QString(g_szFewArgs));
        }
        else
        {
            cerr << qPrintable(QString(g_szManyArgs));
        }
        result = false;
    }
    else
    {
        if (rCmdParams.contains("repeat"))
        {
            if (rCmdParams["repeat"].isEmpty())
            {
                if (rCmdParams.contains("trainset"))
                {
                    if (rCmdParams["trainset"].isEmpty())
                    {
                        cerr << qPrintable(QString(g_szNullVal).arg("trainset"));
                        result = false;
                    }
                    else
                    {
                        result = true;
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szArgIsNotFound).arg("trainset"));
                    result = false;
                }
            }
            else
            {
                result = false;
                cerr<<qPrintable(QString(g_szImpossibleVal).arg("repeat"));
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("repeat"));
            result = false;
        }
    }
    return result;
}

/* В обучающем множестве найти группы противоречивых примеров, у которых
входные сигналы одинаковы, а желаемые выходные - разные.
   Обучающее множество задано последовательностью входных сигналов aTrainInputs
и последовательностью желаемых выходных сигналов aTrainTargets. Количество
входных сигналов соответствует количеству желаемых выходных сигналов и
составляет nSamplesNumber. Размер входного сигнала равен nInputSize, а размер
желаемого выходного сигнала - nOutputSize.
   В результате своей работы функция формирует двухуровневый список
aGroupsOfDivergentSamples, на первом уровне которого - группы противоречивых
примеров, а на втором уровне - индексы противоречивых примеров в обучающем
множестве для каждой из групп. */
void find_divergent_samples_in_train_set(
        const double aTrainInputs[], const double aTrainTargets[],
        int nSamplesNumber, int nInputSize, int nOutputSize,
        QList<QList<int> >& aGroupsOfDivergentSamples)
{
    int i, ind1, ind2;
    QList<QList<int> >::iterator it1, it2;
    QList<int> aNewGroup;

    aGroupsOfDivergentSamples.clear();
    for (i = 0; i < nSamplesNumber; i++)
    {
        aNewGroup.clear();
        aNewGroup.append(i);
        aGroupsOfDivergentSamples.append(aNewGroup);
    }

    it1 = aGroupsOfDivergentSamples.begin();
    while (it1 != aGroupsOfDivergentSamples.end())
    {
        ind1 = (*it1).first();
        it2 = it1 + 1;
        while (it2 != aGroupsOfDivergentSamples.end())
        {
            ind2 = (*it2).first();
            if (same_train_signals(&aTrainInputs[ind1 * nInputSize],
                                   &aTrainInputs[ind2 * nInputSize],
                                   nInputSize))
            {
                if (!same_train_signals(&aTrainTargets[ind1 * nOutputSize],
                                        &aTrainTargets[ind2 * nOutputSize],
                                        nOutputSize))
                {
                    (*it1).append(ind2);
                    it2 = aGroupsOfDivergentSamples.erase(it2);
                }
                else
                {
                    it2++;
                }
            }
            else
            {
                it2++;
            }
        }
        it1++;
    }

    it1 = aGroupsOfDivergentSamples.begin();
    while (it1 != aGroupsOfDivergentSamples.end())
    {
        if ((*it1).size() <= 1)
        {
            it1 = aGroupsOfDivergentSamples.erase(it1);
        }
        else
        {
            it1++;
        }
    }
}

/* В обучающем множестве найти группы повторяющихся примеров, у которых
одинаковы и входные, и желаемые выходные сигналы.
   Обучающее множество задано последовательностью входных сигналов aTrainInputs
и последовательностью желаемых выходных сигналов aTrainTargets. Количество
входных сигналов соответствует количеству желаемых выходных сигналов и
составляет nSamplesNumber. Размер входного сигнала равен nInputSize, а размер
желаемого выходного сигнала - nOutputSize.
   Желаемые выходные сигналы в обучающем множестве могут и отсутствовать
(aTrainTargets является нулевым указателем, а размер одного желаемого выходного
сигнала nOutputSize равен нулю). В этом случае примеры сравниваются только по
своим входным сигналам.
   В результате своей работы функция формирует двухуровневый список
aGroupsOfRepeatingSamples, на первом уровне которого - группы повторяющихся
примеров, а на втором уровне - индексы повторяющихся примеров в обучающем
множестве для каждой из групп. */
void find_repeating_samples_in_train_set(
        const double aTrainInputs[], const double aTrainTargets[],
        int nSamplesNumber, int nInputSize, int nOutputSize,
        QList<QList<int> >& aGroupsOfRepeatingSamples)
{
    int i, ind1, ind2;
    QList<QList<int> >::iterator it1, it2;
    QList<int> aNewGroup;

    aGroupsOfRepeatingSamples.clear();
    for (i = 0; i < nSamplesNumber; i++)
    {
        aNewGroup.clear();
        aNewGroup.append(i);
        aGroupsOfRepeatingSamples.append(aNewGroup);
    }

    it1 = aGroupsOfRepeatingSamples.begin();
    while (it1 != aGroupsOfRepeatingSamples.end())
    {
        ind1 = (*it1).first();
        it2 = it1 + 1;
        while (it2 != aGroupsOfRepeatingSamples.end())
        {
            ind2 = (*it2).first();
            if (same_train_signals(&aTrainInputs[ind1 * nInputSize],
                                   &aTrainInputs[ind2 * nInputSize],
                                   nInputSize))
            {
                if (nOutputSize == 0)
                {
                    (*it1).append(ind2);
                    it2 = aGroupsOfRepeatingSamples.erase(it2);
                }
                else
                {
                    if (same_train_signals(&aTrainTargets[ind1 * nOutputSize],
                                           &aTrainTargets[ind2 * nOutputSize],
                                           nOutputSize))
                    {
                        (*it1).append(ind2);
                        it2 = aGroupsOfRepeatingSamples.erase(it2);
                    }
                    else
                    {
                        it2++;
                    }
                }
            }
            else
            {
                it2++;
            }
        }
        it1++;
    }

    it1 = aGroupsOfRepeatingSamples.begin();
    while (it1 != aGroupsOfRepeatingSamples.end())
    {
        if ((*it1).size() <= 1)
        {
            it1 = aGroupsOfRepeatingSamples.erase(it1);
        }
        else
        {
            it1++;
        }
    }
}

/* Вывести на экран информацию о группах противоречивых примеров обучающего
множества, представленную в двухуровневом списке aGroupsOfDivergentSamples
(на первом уровне этого списка - группы противоречивых примеров, а на втором
уровне - индексы противоречивых примеров в обучающем множестве для каждой из
групп). */
void print_divergent_samples(
        const QList<QList<int> >& aGroupsOfDivergentSamples)
{
    QList<QList<int> >::const_iterator it1;
    QList<int>::const_iterator it2;
    int i, n;
    QString sTmp;

    i = 1;
    for (it1 = aGroupsOfDivergentSamples.begin();
         it1 != aGroupsOfDivergentSamples.end();
         it1++)
    {
        n = (*it1).size();
        sTmp = QString::number(n);
        if (sTmp.size() == 1)
        {
            if (sTmp[0] == '1')
            {
                cout << qPrintable(QString(g_szGroupOfDivergentSamples1).arg(
                                       i).arg(n));
            }
            else if ((sTmp[0] == '2') || (sTmp[0] == '3') || (sTmp[0] == '4'))
            {
                cout << qPrintable(QString(g_szGroupOfDivergentSamples2).arg(
                                       i).arg(n));
            }
            else
            {
                cout << qPrintable(QString(g_szGroupOfDivergentSamples3).arg(
                                       i).arg(n));
            }
        }
        else
        {
            if (sTmp[sTmp.size()-2] == '1')
            {
                cout << qPrintable(QString(g_szGroupOfDivergentSamples3).arg(
                                       i).arg(n));
            }
            else
            {
                if (sTmp[sTmp.size()-1] == '1')
                {
                    cout<<qPrintable(QString(g_szGroupOfDivergentSamples1).arg(
                                         i).arg(n));
                }
                else if ((sTmp[sTmp.size()-1] == '2')
                         || (sTmp[sTmp.size()-1] == '3')
                         || (sTmp[sTmp.size()-1] == '4'))
                {
                    cout<<qPrintable(QString(g_szGroupOfDivergentSamples2).arg(
                                         i).arg(n));
                }
                else
                {
                    cout<<qPrintable(QString(g_szGroupOfDivergentSamples3).arg(
                                         i).arg(n));
                }
            }
        }

        it2 = (*it1).begin();
        cout << ' ' << (*it2);
        it2++;
        while (it2 != (*it1).end())
        {
            cout << ", " << (*it2);
            it2++;
        }
        cout << '.' << endl;

        i++;
    }
}

/* Удалить из обучающего множества все противоречивые примеры и вернуть новое
количество примеров в этом обучающем множестве.
   Обучающее множество задано последовательностью входных сигналов aTrainInputs
и последовательностью желаемых выходных сигналов aTrainTargets. Количество
входных сигналов соответствует количеству желаемых выходных сигналов и
составляет nSamplesNumber. Размер входного сигнала равен nInputSize, а размер
желаемого выходного сигнала - nOutputSize.
   Группы противоречивых примеров заданы двухуровневым списком
aGroupsOfDivergentSamples, на первом уровне которого - группы противоречивых
примеров, а на втором уровне - индексы противоречивых примеров в обучающем
множестве для каждой из групп. */
int remove_divergent_samples(
        double aTrainInputs[], double aTrainTargets[],
        int nSamplesNumber, int nInputSize, int nOutputSize,
        const QList<QList<int> >& aGroupsOfDivergentSamples)
{
    QVector<bool> aRemovedSamples(nSamplesNumber);
    QList<QList<int> >::const_iterator it1;
    QList<int>::const_iterator it2;
    int i, j = 0, res = nSamplesNumber;

    aRemovedSamples.fill(false);
    for (it1 = aGroupsOfDivergentSamples.begin();
         it1 != aGroupsOfDivergentSamples.end();
         it1++)
    {
        for (it2 = (*it1).begin(); it2 != (*it1).end(); it2++)
        {
            aRemovedSamples[*it2] = true;
        }
    }

    for (i = 0; i < nSamplesNumber; i++)
    {
        if (aRemovedSamples[i])
        {
            if (j < (res-1))
            {
                memmove(&aTrainInputs[j*nInputSize],
                        &aTrainInputs[(j+1)*nInputSize],
                        (res-j-1)*nInputSize*sizeof(double));
                memmove(&aTrainTargets[j*nOutputSize],
                        &aTrainTargets[(j+1)*nOutputSize],
                        (res-j-1)*nOutputSize*sizeof(double));
            }
            res--;
        }
        else
        {
            j++;
        }
    }
    return res;
}

/* Удалить из обучающего множества все группы повторяющихся примеров, оставив
только по одному примеру из каждой группы, и вернуть новое количество примеров
в этом обучающем множестве.
   Обучающее множество задано последовательностью входных сигналов aTrainInputs
и последовательностью желаемых выходных сигналов aTrainTargets. Количество
входных сигналов соответствует количеству желаемых выходных сигналов и
составляет nSamplesNumber. Размер входного сигнала равен nInputSize, а размер
желаемого выходного сигнала - nOutputSize.
   Группы повторяющихся примеров заданы двухуровневым списком
aGroupsOfRepeatingSamples, на первом уровне которого - группы повторяющихся
примеров, а на втором уровне - индексы повторяющихся примеров в обучающем
множестве для каждой из групп. */
int remove_repeating_samples(
        double aTrainInputs[], double aTrainTargets[],
        int nSamplesNumber, int nInputSize, int nOutputSize,
        const QList<QList<int> >& aGroupsOfRepeatingSamples)
{
    QVector<bool> aRemovedSamples(nSamplesNumber);
    QList<QList<int> >::const_iterator it1;
    QList<int>::const_iterator it2;
    int i, j = 0, res = nSamplesNumber;

    aRemovedSamples.fill(false);
    for (it1 = aGroupsOfRepeatingSamples.begin();
         it1 != aGroupsOfRepeatingSamples.end();
         it1++)
    {
        it2 = (*it1).begin(); it2++;
        while (it2 != (*it1).end())
        {
            aRemovedSamples[*it2] = true;
            it2++;
        }
    }

    for (i = 0; i < nSamplesNumber; i++)
    {
        if (aRemovedSamples[i])
        {
            if (j < (res-1))
            {
                memmove(&aTrainInputs[j*nInputSize],
                        &aTrainInputs[(j+1)*nInputSize],
                        (res-j-1)*nInputSize*sizeof(double));
                memmove(&aTrainTargets[j*nOutputSize],
                        &aTrainTargets[(j+1)*nOutputSize],
                        (res-j-1)*nOutputSize*sizeof(double));
            }
            res--;
        }
        else
        {
            j++;
        }
    }
    return res;
}

int unite_divergent_samples(
        double aTrainInputs[], double aTrainTargets[],
        int nSamplesNumber, int nInputSize, int nOutputSize,
        const QList<QList<int> >& aGroupsOfDivergentSamples,
        TSolvedTask task)
{
    QVector<bool> aRemovedSamples(nSamplesNumber);
    QList<QList<int> >::const_iterator it1;
    QList<int>::const_iterator it2, it3;
    int i, j, n, iMax, res;
    QVector<double> aTempOutput(nOutputSize);

    aRemovedSamples.fill(false);
    for (it1 = aGroupsOfDivergentSamples.begin();
         it1 != aGroupsOfDivergentSamples.end();
         it1++)
    {
        it2 = (*it1).begin();
        n = 1;

        if ((task == taskREGRESSION)
                || ((task == taskCLASSIFICATION) && (nOutputSize == 1)))
        {
            for (it3 = it2+1; it3 != (*it1).end(); it3++)
            {
                for (i = 0; i < nOutputSize; i++)
                {
                    aTrainTargets[(*it2) * nOutputSize + i]
                            += aTrainTargets[(*it3) * nOutputSize + i];
                }
                n++;
                aRemovedSamples[*it3] = true;
            }

            for (i = 0; i < nInputSize; i++)
            {
                aTrainInputs[(*it2) * nInputSize + i] /= n;
            }
            for (i = 0; i < nOutputSize; i++)
            {
                aTrainTargets[(*it2) * nOutputSize + i] /= n;
            }
        }
        else
        {
            aTempOutput.fill(-1.0);
            iMax = find_maximum_component(
                        &aTrainTargets[(*it2) * nOutputSize], nOutputSize);
            aTempOutput[iMax] = 1.0;
            for (it3 = it2+1; it3 != (*it1).end(); it3++)
            {
                iMax = find_maximum_component(
                            &aTrainTargets[(*it3) * nOutputSize], nOutputSize);
                aTempOutput[iMax] = 1.0;
                aRemovedSamples[*it3] = true;
            }
            for (i = 0; i < nOutputSize; i++)
            {
                aTrainTargets[(*it2) * nOutputSize + i] = aTempOutput[i];
            }
        }
    }

    j = 0; res = nSamplesNumber;
    for (i = 0; i < nSamplesNumber; i++)
    {
        if (aRemovedSamples[i])
        {
            if (j < (res-1))
            {
                memmove(&aTrainInputs[j*nInputSize],
                        &aTrainInputs[(j+1)*nInputSize],
                       (res-j-1)*nInputSize*sizeof(double));
                memmove(&aTrainTargets[j*nOutputSize],
                        &aTrainTargets[(j+1)*nOutputSize],
                        (res-j-1)*nOutputSize*sizeof(double));
            }
            res--;
        }
        else
        {
            j++;
        }
    }
    return res;
}

/* Вывести на экран временные характеристики (суммарное время обработки
множества входных сигналов и среднее время обработки одного входного сигнала)
работы нейросети.
   Суммарное время обработки множества входных сигналов (в секундах) задано
входным аргументом time_at_secs, а количество входных сигналов тестового
множества - входным аргументом nSamplesNumber. */
void print_timing_performances(const double& time_at_secs, int nSamplesNumber)
{
    QString sInformation;
    cout.setf(ios_base::fixed, ios_base::floatfield);

    cout.precision(3);
    sInformation = QString(g_szAllInputsProcessingDuration).arg(time_at_secs);
    cout << qPrintable(sInformation) << endl;

    double average_time = (1000.0 * time_at_secs) / nSamplesNumber;
    if (average_time >= 1000.0)
    {
        average_time = time_at_secs / nSamplesNumber;
        sInformation = QString(g_szOneInputProcessingDurationAtSecs).arg(
                    average_time);
    }
    else
    {
        sInformation = QString(g_szOneInputProcessingDurationAtMsecs).arg(
                    average_time);
    }
    cout << qPrintable(sInformation) << endl;
}

/* Прочитать из текстового CSV-файла sCSVFile матрицу вещественных чисел
aCSVData. При этом предполагается, что длина каждой строки матрицы равна
nSizeOfCSVRow. Вернуть количество прочитанных строк, если чтение завершилось
успехом, и ноль - в случае неудачи, вызванной какой-либо ошибкой. */
int readCSV(const QString& sCSVFile, int nSizeOfCSVRow,
            QList<QVector<double> >& aCSVData)
{
    int result = 0;
    QFile csvFile(sCSVFile);

    aCSVData.clear();
    if (!csvFile.open(QFile::ReadOnly | QFile::Text))
    {
        return 0;
    }

    QTextStream csvStream(&csvFile);
    QString sReadLine;
    QStringList aLineParts;
    QStringList::iterator it;
    QVector<double> aCSVRow(nSizeOfCSVRow);
    bool ok = true;
    int i;

    while ((csvStream.status() == QTextStream::Ok) && (!csvStream.atEnd()))
    {
        sReadLine = csvStream.readLine().trimmed();
        if (sReadLine.isEmpty())
        {
            continue;
        }
        aLineParts = sReadLine.split(',', QString::SkipEmptyParts);
        if (aLineParts.size() != nSizeOfCSVRow)
        {
            result = 0;
            break;
        }
        it = aLineParts.begin();
        for (i = 0; i < nSizeOfCSVRow; i++)
        {
            aCSVRow[i] = (*it).trimmed().toFloat(&ok);
            if (!ok)
            {
                break;
            }
            it++;
        }
        if (!ok)
        {
            result = 0;
            break;
        }
        aCSVData.append(aCSVRow);
        result++;
    }
    if ((result < 1) || (!((csvStream.status() == QTextStream::ReadPastEnd)
                           || csvStream.atEnd())))
    {
        aCSVData.clear();
        return 0;
    }
    return result;
}

/*****************************************************************************/
/* ВНЕШНИЕ ФУНКЦИИ МОДУЛЯ commands_unit.cpp */
/*****************************************************************************/

/* Определить, какой из режимов выполнения программы выбран:
   1) изменение структуры нейросети;
   2) вывод на экран структуры нейросети;
   3) изменение веса (весов) нейросети;
   4) вывод на экран веса (весов) нейросети;
   5) инициализация весов нейросети случайными значениями;
   6) обучение нейросети;
   7) использование нейросети.
   Определение режима выполнения происходит путём поиска соответствующего
ключа в списке ключей запуска rCmdParams, сформированном путём анализа
аргументов командной строки (см. описание функции parse_command_line). */
TExecutionMode detect_mode(const TCmdParams& rCmdParams)
{
    TExecutionMode result = UNKNOWN_MODE;
    if (rCmdParams.contains("mlp"))
    {
        if (rCmdParams.contains("set"))
        {
            if (rCmdParams.contains("struct"))
            {
                result = SET_MLP_STRUCTURE;
            }
            else
            {
                if (rCmdParams.contains("w"))
                {
                    result = SET_MLP_WEIGHT;
                }
            }
        }
        else
        {
            if (rCmdParams.contains("get"))
            {
                if (rCmdParams.contains("struct"))
                {
                    result = GET_MLP_STRUCTURE;
                }
                else
                {
                    if (rCmdParams.contains("w"))
                    {
                        result = GET_MLP_WEIGHT;
                    }
                }
            }
            else
            {
                if (rCmdParams.contains("init"))
                {
                    result = INITIALIZE_MLP;
                }
                else
                {
                    if (rCmdParams.contains("train"))
                    {
                        result = TRAIN_MLP;
                    }
                    else
                    {
                        if (rCmdParams.contains("in"))
                        {
                            result = USE_MLP;
                        }
                    }
                }
            }
        }
    }
    else
    {
        if (rCmdParams.contains("trainset"))
        {
            if (rCmdParams.contains("csv"))
            {
                if (rCmdParams.contains("to_ts"))
                {
                    result = CSV_TO_TRAINSET;
                }
                else if (rCmdParams.contains("to_csv"))
                {
                    result = TRAINSET_TO_CSV;
                }
            }
            else
            {
                if (rCmdParams.contains("controlset"))
                {
                    result = SEPARATE_TRAINSET;
                }
                else
                {
                    if (rCmdParams.contains("divergent"))
                    {
                        result = PROCESS_DIVERGENT_SAMPLES;
                    }
                    else
                    {
                        if (rCmdParams.contains("repeat"))
                        {
                            result = REMOVE_REPEATING_SAMPLES;
                        }
                        else
                        {
                            result = SHOW_TRAINSET;
                        }
                    }
                }
            }
        }
    }
    return result;
}

/* Установить структуру нейронной сети, загруженной из заданного файла.
Описание структуры представляет собой строку в виде перечня целых
положительных чисел, разделённых символами "тире":
  <целое число>-<целое число>-...-<целое число>
  Первое целое число - это количество входов в нейросеть, а остальные целые
числа со второго по последнее - это размеры слоёв нейросети с первого
по последний соответственно.
  ВХОДНЫЕ АРГУМЕНТЫ
  rCmpParams - список параметров командной строки в виде пары "ключ-значение".
Для данной функции (установки структуры нейронной сети) в списке должно быть
три ключа: "mlp", "set" и "struct". Значение ключа "mlp" определяет название
файла, а значение ключа "set" - строку с описанием новой структуры.
Ключ "struct" должен быть указан без значения, поскольку сам факт наличия этого
ключа указывает, что работа будет происходить со структурой нейронной сети.
  ВОЗВРАЩАЕМЫЙ РЕЗУЛЬТАТ
  В случае успешного завершения своей работы функция возвращает true. В случае
ошибки (например, список rCmdParams некорректен, либо не удалось создать файл
с новой нейронной сетью, либо файл с нейросетью существует, но не содержит
нейросеть, и т.п.) возвращается false, а на экран выводится сообщение об
ошибке.
  ПРИМЕЧАНИЕ
  Если заданный файл с нейросетью не существует, то он будет создан, и в него
будет записана новая созданная нейросеть. Если же файл существует и уже
содержит нейросеть, то структура этой сети будет изменена. */
bool setStructureOfMLP(const TCmdParams& rCmdParams)
{
    bool result = check_params_for_setStructureOfMLP(rCmdParams);

    if (result)
    {
        int *aLayerSizes = NULL;
        TActivationKind *aActivations = NULL;
        try
        {
            int nInputsCount, nLayersCount;
            if (parse_mlp_structure_description(rCmdParams["set"],nInputsCount,
                                                nLayersCount, NULL, NULL))
            {
                aLayerSizes = new int[nLayersCount];
                aActivations = new TActivationKind[nLayersCount];
                parse_mlp_structure_description(rCmdParams["set"],nInputsCount,
                                                nLayersCount, aLayerSizes,
                                                aActivations);

                CMultilayerPerceptron changed_mlp;
                QString sFilename = rCmdParams["mlp"];
                if (QFile::exists(sFilename))
                {
                    if (changed_mlp.load(sFilename))
                    {
                        int i, j, k, nNeuronsCount;
                        int nOldInputsCount, nNewInputsCount;
                        double value;
                        CMultilayerPerceptron new_mlp(nInputsCount,
                                                      nLayersCount,
                                                      aLayerSizes,
                                                      aActivations);
                        new_mlp.initialize_weights();

                        if (new_mlp.getLayersCount()
                            > changed_mlp.getLayersCount())
                        {
                            nLayersCount = changed_mlp.getLayersCount();
                        }
                        else
                        {
                            nLayersCount = new_mlp.getLayersCount();
                        }
                        for (i = 0; i < nLayersCount; i++)
                        {
                            if (new_mlp.getLayerSize(i)
                                > changed_mlp.getLayerSize(i))
                            {
                                nNeuronsCount = changed_mlp.getLayerSize(i);
                            }
                            else
                            {
                                nNeuronsCount = new_mlp.getLayerSize(i);
                            }
                            nOldInputsCount
                                    = changed_mlp.getInputsCountOfLayer(i);
                            nNewInputsCount = new_mlp.getInputsCountOfLayer(i);
                            if (nNewInputsCount > nOldInputsCount)
                            {
                                nInputsCount = nOldInputsCount;
                            }
                            else
                            {
                                nInputsCount = nNewInputsCount;
                            }
                            for (j = 0; j < nNeuronsCount; j++)
                            {
                                for (k = 0; k < nInputsCount; k++)
                                {
                                    value = changed_mlp.getWeight(i, j, k);
                                    new_mlp.setWeight(i, j, k, value);
                                }
                                value = changed_mlp.getWeight(i,j,
                                                              nOldInputsCount);
                                new_mlp.setWeight(i,j,nNewInputsCount, value);
                            }
                        }
                        changed_mlp = new_mlp;
                        result = true;
                    }
                    else
                    {
                        cerr << qPrintable(QString(g_szMlpReadingError).arg(
                                               sFilename));
                    }
                }
                else
                {
                    result = true;
                    changed_mlp.resize(nInputsCount, nLayersCount,
                                       aLayerSizes, aActivations);
                }
                if (result)
                {
                    if (!changed_mlp.save(sFilename))
                    {
                        result = false;
                        cerr << qPrintable(QString(g_szMlpWritingError).arg(
                                               sFilename));
                    }
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szMlpStructureError));
                result = false;
            }            
            if (aLayerSizes != NULL)
            {
                delete[] aLayerSizes;
                aLayerSizes = NULL;
            }
            if (aActivations != NULL)
            {
                delete[] aActivations;
                aActivations = NULL;
            }
        }
        catch(...)
        {
            if (aLayerSizes != NULL)
            {
                delete[] aLayerSizes;
                aLayerSizes = NULL;
            }
            if (aActivations != NULL)
            {
                delete[] aActivations;
                aActivations = NULL;
            }
            throw;
        }
    }
    return result;
}

/* Вывести на экран описание структуры нейросети из, загруженной из заданного
файла. В случае успешного завершения возвращается true, в случае ошибки
- false. Также в случае ошибки происходит вывод на экран соответствующего
сообщения.
  ВХОДНЫЕ АРГУМЕНТЫ
  rCmpParams - список параметров командной строки в виде пары "ключ-значение".
Для данной функции (получения информации о структурЕ нейронной сети) в списке
должно быть три ключа: "mlp", "get" и "struct". Значение ключа "mlp" определяет
название файла, а ключи "get" и "struct" должны быть указаны без значений,
поскольку факт наличия ключа "struct" показывает, что работа будет происходить
со структурой нейронной сети, а факт наличия ключа "get" - что информацию
об этой структуре нам необходимо вывести на экран (в поток stdout).
  ВОЗВРАЩАЕМЫЙ РЕЗУЛЬТАТ
  В случае успешного завершения своей работы функция возвращает true. В случае
ошибки (например, список rCmdParams некорректен, либо указанный файл с
нейронной сетью не существует, либо этот файл существует, но не содержит
нейросеть, и т.п.) возвращается false, а на экран выводится сообщение об
ошибке. */
bool getStructureOfMLP(const TCmdParams& rCmdParams)
{
    if (!check_params_for_getStructureOfMLP(rCmdParams))
    {
        return false;
    }

    QString sFilename = rCmdParams["mlp"];
    if (!QFile::exists(sFilename))
    {
        cerr << qPrintable(QString(g_szFileDoesNotExist).arg(sFilename));
        return false;
    }
    CMultilayerPerceptron loaded_mlp;
    if (!loaded_mlp.load(sFilename))
    {
        cerr << qPrintable(QString(g_szMlpReadingError).arg(sFilename));
        return false;
    }
    cout << qPrintable(QString(g_szTotalWeightsNumber).arg(
                           loaded_mlp.getAllWeightsCount())) << endl;
    cout << qPrintable(QString(g_szInputsNumber).arg(
                           loaded_mlp.getInputsCount())) << endl;
    cout << qPrintable(QString(g_szLayersNumber).arg(
                           loaded_mlp.getLayersCount())) << endl;

    for (int i = 0; i < loaded_mlp.getLayersCount(); i++)
    {
        cout << qPrintable(QString(g_szLayerNumber).arg(i+1)) << ": "
             << qPrintable(layer_description_to_string(loaded_mlp, i))
             << endl;
    }
    return true;
}

/* Установить значение веса или группы весов заданной нейронной сети.
   ВХОДНЫЕ АРГУМЕНТЫ
   rCmpParams - список параметров командной строки в виде пары "ключ-значение".
Для данной функции (получения информации о структуре нейронной сети) в списке
должно быть три ключа: "mlp", "set" и "w".
   1. "mlp" - ключ, значением которого является название файла с нейронной
сетью.
   2. "w" - ключ, указывающий, что операция будет проводится над весом или
группой весов нейронной сети. Значением ключа является строка, содержащая
индексы изменяемого веса (весов) в виде последовательности неотрицательных
целых чисел, разделённых символом "тире".
   Если данная строка пуста, то изменяются значения всех весов сети.
   Если строка содержит только одно число, то это число - номер слоя, все веса
нейронов которого будут изменены.
   Если строка содержит два числа, то эти числа - номера слоя и нейрона в слое
соответственно. Всем весам этого нейрона будет присвоено новое значение.
   Если строка содержит три числа, то эти числа - номера слоя, нейрона в слое
и входа в нейрон соответственно. Вес указанной связи будет изменён.
   3. "set" - ключ, указывающий тип операции над весом (весами) нейросети -
установка новых величин весов. Значением этого ключа является строка,
содержащая произвольное вещественное число - новую величину веса (группы
весов).
   ВОЗВРАЩАЕМЫЙ РЕЗУЛЬТАТ
   Если функция успешно завершила свою работу, то возвращается true. В случае
ошибки возвращается false и на экран выводится сообщение о соответствующей
ошибке. */
bool setWeightOfMLP(const TCmdParams& rCmdParams)
{
    if (!check_params_for_setWeightOfMLP(rCmdParams))
    {
        return false;
    }

    QString sMlpFilename = rCmdParams["mlp"];
    if (!QFile::exists(sMlpFilename))
    {
        cerr << qPrintable(QString(g_szFileDoesNotExist).arg(sMlpFilename));
        return false;
    }

    CMultilayerPerceptron changed_mlp;
    if (!changed_mlp.load(sMlpFilename))
    {
        cerr << qPrintable(QString(g_szMlpReadingError).arg(sMlpFilename));
        return false;
    }

    bool result;
    int nIndexesCount = 0;
    int *aIndexes = NULL;
    try
    {
        nIndexesCount = str_to_int_array(rCmdParams["w"], aIndexes);
        if ((nIndexesCount > 0) && (nIndexesCount <= 3))
        {
            aIndexes = new int[nIndexesCount];
            str_to_int_array(rCmdParams["w"], aIndexes);
        }
        result = check_mlp_indexes(aIndexes, nIndexesCount, changed_mlp);
        if (result)
        {
            bool ok = true;
            double newval;
            int i, j, k, nNeuronsCount, nInputsCount;
            newval = rCmdParams["set"].toFloat(&ok);
            if (ok)
            {
                switch (nIndexesCount)
                {
                case 0:
                    for (i = 0; i < changed_mlp.getLayersCount(); i++)
                    {
                        nNeuronsCount = changed_mlp.getLayerSize(i);
                        nInputsCount = changed_mlp.getInputsCountOfLayer(i);
                        for (j = 0; j < nNeuronsCount; j++)
                        {
                            for (k = 0; k <= nInputsCount; k++)
                            {
                                changed_mlp.setWeight(i, j, k, newval);
                            }
                        }
                    }
                    break;
                case 1:
                    i = aIndexes[0];
                    nNeuronsCount = changed_mlp.getLayerSize(i);
                    nInputsCount = changed_mlp.getInputsCountOfLayer(i);
                    for (j = 0; j < nNeuronsCount; j++)
                    {
                        for (k = 0; k <= nInputsCount; k++)
                        {
                            changed_mlp.setWeight(i, j, k, newval);
                        }
                    }
                    break;
                case 2:
                    i = aIndexes[0];
                    j = aIndexes[1];
                    nInputsCount = changed_mlp.getInputsCountOfLayer(i);
                    for (k = 0; k <= nInputsCount; k++)
                    {
                        changed_mlp.setWeight(i, j, k, newval);
                    }
                    break;
                case 3:
                    i = aIndexes[0];
                    j = aIndexes[1];
                    k = aIndexes[2];
                    changed_mlp.setWeight(i, j, k, newval);
                }
                if (!changed_mlp.save(sMlpFilename))
                {
                    cerr << qPrintable(QString(g_szMlpWritingError).arg(
                                           sMlpFilename));
                    result = false;
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szIncorrectWeight));
                result = false;
            }
        }
        if (aIndexes != NULL)
        {
            delete[] aIndexes;
        }
    }
    catch(...)
    {
        if (aIndexes != NULL)
        {
            delete[] aIndexes;
        }
        throw;
    }
    return result;
}

/* Вывести на экран значение заданного веса либо значения заданной группы весов
нейросети.
   ВХОДНЫЕ АРГУМЕНТЫ
   rCmpParams - список параметров командной строки в виде пары "ключ-значение".
Для данной функции (получения информации о структуре нейронной сети) в списке
должно быть три ключа: "mlp", "set" и "w".
   1. "mlp" - ключ, значением которого является название файла с нейронной
сетью.
   2. "w" - ключ, указывающий, что операция будет проводится над весом или
группой весов нейронной сети. Значением ключа является строка, содержащая
индексы изменяемого веса (весов) в виде последовательности неотрицательных
целых чисел, разделённых символом "тире".
   Если данная строка пуста, то изменяются значения всех весов сети.
   Если строка содержит только одно число, то это число - номер слоя, все веса
нейронов которого будут изменены.
   Если строка содержит два числа, то эти числа - номера слоя и нейрона в слое
соответственно. Всем весам этого нейрона будет присвоено новое значение.
   Если строка содержит три числа, то эти числа - номера слоя, нейрона в слое
и входа в нейрон соответственно. Вес указанной связи будет изменён.
   3. "get" - ключ, указывающий тип операции над весом (весами) нейросети -
вывод информации о значениях весов нейросети. Ключ должен быть указан без
значения.
   ВОЗВРАЩАЕМЫЙ РЕЗУЛЬТАТ
   Если функция успешно завершила свою работу, то возвращается true. В случае
ошибки возвращается false и на экран выводится сообщение о соответствующей
ошибке. */
bool getWeightOfMLP(const TCmdParams& rCmdParams)
{
    if (!check_params_for_getWeightOfMLP(rCmdParams))
    {
        return false;
    }

    QString sMlpFilename = rCmdParams["mlp"];
    if (!QFile::exists(sMlpFilename))
    {
        cerr << qPrintable(QString(g_szFileDoesNotExist).arg(sMlpFilename));
        return false;
    }

    CMultilayerPerceptron loaded_mlp;
    if (!loaded_mlp.load(sMlpFilename))
    {
        cerr << qPrintable(QString(g_szMlpReadingError).arg(sMlpFilename));
        return false;
    }

    bool result;
    int nIndexesCount = 0;
    int *aIndexes = NULL;
    try
    {
        nIndexesCount = str_to_int_array(rCmdParams["w"], aIndexes);
        if ((nIndexesCount > 0) && (nIndexesCount <= 3))
        {
            aIndexes = new int[nIndexesCount];
            str_to_int_array(rCmdParams["w"], aIndexes);
        }
        result = check_mlp_indexes(aIndexes, nIndexesCount, loaded_mlp);
        if (result)
        {
            int i, j, k, nNeuronsCount, nInputsCount;
            switch (nIndexesCount)
            {
            case 0:
                for (i = 0; i < loaded_mlp.getLayersCount(); i++)
                {
                    cout << qPrintable(QString(g_szLayerName).arg(i+1))
                         << endl;
                    nNeuronsCount = loaded_mlp.getLayerSize(i);
                    nInputsCount = loaded_mlp.getInputsCountOfLayer(i);
                    for (j = 0; j < nNeuronsCount; j++)
                    {
                        cout << qPrintable(QString(g_szNeuronName).arg(j+1))
                             << endl;
                        for (k = 0; k < nInputsCount; k++)
                        {
                            cout <<qPrintable(QString(g_szWeightName).arg(k+1))
                                 <<" "<<loaded_mlp.getWeight(i, j, k) << endl;
                        }
                        cout << qPrintable(QString(g_szBiasName))
                             << " " << loaded_mlp.getWeight(i, j, k) << endl;
                    }
                }
                break;
            case 1:
                i = aIndexes[0];
                nNeuronsCount = loaded_mlp.getLayerSize(i);
                nInputsCount = loaded_mlp.getInputsCountOfLayer(i);
                for (j = 0; j < nNeuronsCount; j++)
                {
                    cout << qPrintable(QString(g_szNeuronName).arg(j+1))
                         << endl;
                    for (k = 0; k < nInputsCount; k++)
                    {
                        cout << qPrintable(QString(g_szWeightName).arg(k+1))
                             << " " << loaded_mlp.getWeight(i, j, k) << endl;
                    }
                    cout << qPrintable(QString(g_szBiasName))
                         << " " << loaded_mlp.getWeight(i, j, k) << endl;
                }
                break;
            case 2:
                i = aIndexes[0];
                j = aIndexes[1];
                nInputsCount = loaded_mlp.getInputsCountOfLayer(i);
                for (k = 0; k < nInputsCount; k++)
                {
                    cout << qPrintable(QString(g_szWeightName).arg(k+1))
                         << " " << loaded_mlp.getWeight(i, j, k) << endl;
                }
                cout << qPrintable(QString(g_szBiasName))
                     << " " << loaded_mlp.getWeight(i, j, k) << endl;
                break;
            case 3:
                i = aIndexes[0];
                j = aIndexes[1];
                k = aIndexes[2];
                cout << loaded_mlp.getWeight(i, j, k) << endl;
            }
        }
        if (aIndexes != NULL)
        {
            delete[] aIndexes;
        }
    }
    catch(...)
    {
        if (aIndexes != NULL)
        {
            delete[] aIndexes;
        }
        throw;
    }
    return result;
}

/* Инициализировать веса нейросети из заданного файла случайными значениями.
Если функция успешно завершила свою работу, то возвращается true. В случае
ошибки возвращается false и на экран выводится сообщение о соответствующей
ошибке.
   ВХОДНЫЕ АРГУМЕНТЫ
   rCmpParams - список параметров командной строки в виде пары "ключ-значение".
Для данной функции (получения информации о структуре нейронной сети) в списке
должно быть два ключа: "mlp" (название файла с нейронной сетью) и "init" (без
значения).
   ВОЗВРАЩАЕМОЕ ЗНАЧЕНИЕ
   Если функция успешно завершила свою работу, то возвращается true. В случае
ошибки возвращается false и на экран выводится сообщение о соответствующей
ошибке. */
bool initializeMLP(const TCmdParams& rCmdParams)
{
    if (!check_params_for_InitializeMLP(rCmdParams))
    {
        return false;
    }

    QString sMlpFilename = rCmdParams["mlp"];

    if (!QFile::exists(sMlpFilename))
    {
        cerr << qPrintable(QString(g_szFileDoesNotExist).arg(sMlpFilename));
        return false;
    }

    CMultilayerPerceptron initialized_mlp;
    if (!initialized_mlp.load(sMlpFilename))
    {
        cerr << qPrintable(QString(g_szMlpReadingError).arg(sMlpFilename));
        return false;
    }

    initialized_mlp.initialize_weights();

    if (initialized_mlp.save(sMlpFilename))
    {
        return true;
    }
    else
    {
        cerr << qPrintable(QString(g_szMlpWritingError).arg(sMlpFilename));
        return false;
    }
}

/* "Воспользоваться" нейронной сетью, т.е. вычислить отклики нейронной сети
на заданную последовательность входных сигналов.
   ВХОДНЫЕ АРГУМЕНТЫ
   rCmpParams - список параметров командной строки в виде пары "ключ-значение".
Для данной функции (получения информации о структуре нейронной сети) в списке
должно быть три ключа: "mlp", "in" и "out":
   1. "mlp" - ключ, значением которого является строка с названием файла,
содержащего используемую нейронную сеть.
   2. "in" - ключ, значением которого является строка с названием входного
файла данных, содержащего множество входных сигналов для нейронной сети. Если
данный файл содержит и множество желаемых выходных сигналов, то после
вычисления реальных откликов нейросети происходит вычисление её ошибки как
суммы квадратов разностей между реальными и желаемыми выходными сигналами.
   3. "out" - ключ, значением которого является строка с названием выходного
файла данных, предназначенного для записи выходных сигналов нейросети,
вычисленных как отклики на соответствующие входные сигналы. Если этот файл
не существует, то он будет создан, а если существует, то - перезаписан.
   ВОЗВРАЩАЕМЫЙ РЕЗУЛЬТАТ
   Если функция успешно завершила свою работу, то возвращается true. В случае
ошибки возвращается false и на экран выводится сообщение о соответствующей
ошибке. */
bool useMLP(const TCmdParams& rCmdParams)
{
    if (!check_params_for_UseMLP(rCmdParams))
    {
        return false;
    }

    if (!QFile::exists(rCmdParams["mlp"]))
    {
        cerr << qPrintable(QString(g_szFileDoesNotExist).arg(
                               rCmdParams["mlp"]));
        return false;
    }

    QString sInputSet = rCmdParams["in"];

    if (!QFile::exists(sInputSet))
    {
        cerr << qPrintable(QString(g_szFileDoesNotExist).arg(sInputSet));
        return false;
    }

    CMultilayerPerceptron used_mlp;
    if (!used_mlp.load(rCmdParams["mlp"]))
    {
        cerr<<qPrintable(QString(g_szMlpReadingError).arg(rCmdParams["mlp"]));
        return false;
    }

    double *aTestInputs = 0, *aTestTargets = 0, *aResultInputs = 0;
    int nTestSamples, nTestInputs, nTestTargets;
    if (!load_trainset(sInputSet, 0, 0, nTestSamples,nTestInputs,nTestTargets))
    {
        cerr << qPrintable(QString(g_szDatasetReadingError).arg(sInputSet));
        return false;
    }

    int iLastLayer = used_mlp.getLayersCount() - 1;
    int nOutputsCount = used_mlp.getLayerSize(iLastLayer);
    if (((nTestTargets != 0) && (nTestTargets != nOutputsCount))
        || (nTestInputs != used_mlp.getInputsCount()))
    {
        cerr << qPrintable(QString(g_szDatasetStructureError));
        return false;
    }

    bool result = true;
    try
    {
        bool bDurationIsCalculated = false;
        time_t start_time = 0, end_time = 0;

        aTestInputs = new double[nTestSamples * nTestInputs];
        if (nTestTargets > 0)
        {
            aTestTargets = new double[nTestSamples * nTestTargets];
        }
        if (!load_trainset(sInputSet, aTestInputs, aTestTargets, nTestSamples,
                           nTestInputs, nTestTargets))
        {
            cerr<<qPrintable(QString(g_szDatasetReadingError).arg(sInputSet));
            result = false;
        }

        if (result && rCmdParams.contains("out"))
        {
            aResultInputs = new double[nTestSamples * nOutputsCount];
        }
        if (result && (nTestTargets > 0))
        {
            double error;
            if (rCmdParams.contains("task"))
            {
                if (rCmdParams["task"].compare(
                            g_szClassificationTask, Qt::CaseInsensitive) == 0)
                {
                    bDurationIsCalculated = true;
                    start_time = time(0);
                    error = used_mlp.calculate_error(
                            aTestInputs, aTestTargets, nTestSamples,
                            taskCLASSIFICATION);
                    end_time = time(0);
                    cout << qPrintable(QString(g_szClassificationError).arg(
                                           error)) << endl;
                }
                else if (rCmdParams["task"].compare(
                             g_szRegressionTask, Qt::CaseInsensitive) == 0)
                {
                    bDurationIsCalculated = true;
                    start_time = time(0);
                    error = used_mlp.calculate_error(
                            aTestInputs, aTestTargets, nTestSamples,
                            taskREGRESSION);
                    end_time = time(0);
                    cout << qPrintable(QString(g_szRegressionError).arg(error))
                         << endl;
                }
                else
                {
                    result = false;
                    cerr << qPrintable(g_szIncorrectTask);
                }
                if (result)
                {
                    error = used_mlp.calculate_mse(aTestInputs, aTestTargets,
                                                   nTestSamples);
                    cout << qPrintable(QString(g_szMeanSquareError).arg(error))
                         << endl;
                }
            }
            else
            {
                bDurationIsCalculated = true;
                start_time = time(0);
                error = used_mlp.calculate_mse(aTestInputs, aTestTargets,
                                               nTestSamples);
                end_time = time(0);
                cout << qPrintable(QString(g_szMeanSquareError).arg(error))
                     << endl;
            }
        }

        if (result && (aResultInputs != 0))
        {
            if (bDurationIsCalculated)
            {
                used_mlp.calculate_outputs(aTestInputs, aResultInputs,
                                           nTestSamples);
            }
            else
            {
                bDurationIsCalculated = true;
                start_time = time(0);
                used_mlp.calculate_outputs(aTestInputs, aResultInputs,
                                           nTestSamples);
                end_time = time(0);
            }
            if (!save_trainset(rCmdParams["out"], aResultInputs, 0,
                               nTestSamples, nOutputsCount, 0))
            {
                result = false;
                cerr << qPrintable(QString(g_szDatasetWritingError).arg(
                                       rCmdParams["out"]));
            }
        }

        if (result && bDurationIsCalculated)
        {
            print_timing_performances(difftime(end_time, start_time),
                                      nTestSamples);
        }

        if (aTestInputs != 0)
        {
            delete[] aTestInputs;
            aTestInputs = 0;
        }
        if (aTestTargets != 0)
        {
            delete[] aTestTargets;
            aTestTargets = 0;
        }
        if (aResultInputs != 0)
        {
            delete[] aResultInputs;
            aResultInputs = 0;
        }
    }
    catch(...)
    {
        if (aTestInputs != 0)
        {
            delete[] aTestInputs;
            aTestInputs = 0;
        }
        if (aTestTargets != 0)
        {
            delete[] aTestTargets;
            aTestTargets = 0;
        }
        if (aResultInputs != 0)
        {
            delete[] aResultInputs;
            aResultInputs = 0;
        }
        throw;
    }

    return result;
}

/* Вывести на экран информацию о структуре заданного обучающего множества
(количество примеров, размеры входных и желаемых выходных сигналов).
   ВХОДНЫЕ АРГУМЕНТЫ
   rCmpParams - список параметров командной строки в виде пары "ключ-значение".
Для данной функции (получения информации о структуре нейронной сети) в списке
должен быть ключ "trainset", значением которого является строка с названием
файла, содержащего проверяемое обучающее множество.

   ВОЗВРАЩАЕМЫЙ РЕЗУЛЬТАТ
   Если функция успешно завершила свою работу, то возвращается true. В случае
ошибки возвращается false и на экран выводится сообщение о соответствующей
ошибке. */
bool showTrainset(const TCmdParams& rCmdParams)
{
    if (!check_params_for_ShowTrainset(rCmdParams))
    {
        return false;
    }

    if (!QFile::exists(rCmdParams["trainset"]))
    {
        cerr << qPrintable(QString(g_szFileDoesNotExist).arg(
                               rCmdParams["trainset"]));
        return false;
    }

    int nSamplesCount = 0, nInputsCount = 0, nTargetsCount = 0;
    if (!load_trainset(rCmdParams["trainset"], 0, 0, nSamplesCount,
                       nInputsCount, nTargetsCount))
    {
        cerr << qPrintable(QString(g_szTrainsetReadingError).arg(
                               rCmdParams["trainset"]));
        return false;
    }

    cout << qPrintable(QString(g_szTrainSamplesNumber).arg(nSamplesCount))
         << endl;
    cout << qPrintable(QString(g_szTrainInputsNumber).arg(nInputsCount))
         << endl;
    cout << qPrintable(QString(g_szTrainTargetsNumber).arg(nTargetsCount))
         << endl;

    return true;
}

/* Разделить исходное обучающее множество на собственно обучающее и контрольное
подмножества.
   ВХОДНЫЕ АРГУМЕНТЫ
   rCmpParams - список параметров командной строки в виде пары "ключ-значение".
Для данной функции (разделения исходного обучающего множества на собственно
обучающее и контрольное подмножества) в списке должно быть три ключа:
"trainset", "controlset" и "r":
   1. "trainset" - ключ, значением которого является строка с названием файла,
содержащего исходное обучающее множество. В результате работы функции этот файл
перезапишется, и в него будет сохранено выделенное обучающее подмножество
исходного множества.
   2. "controlset" - ключ, значением которого является строка с названием файла,
в который будет записано выделенное контрольное (тестовое) множество.
   3. "r" - коэффициент разбиения, определяющий долю примеров, которые отойдут
в тестовое множество.
   ВОЗВРАЩАЕМЫЙ РЕЗУЛЬТАТ
   Если функция успешно завершила свою работу, то возвращается true. В случае
ошибки возвращается false и на экран выводится сообщение о соответствующей
ошибке. */
bool separateTrainset(const TCmdParams& rCmdParams)
{
    if (!check_params_for_SeparateTrainset(rCmdParams))
    {
        return false;
    }

    if (!QFile::exists(rCmdParams["trainset"]))
    {
        cerr << qPrintable(QString(g_szFileDoesNotExist).arg(
                               rCmdParams["trainset"]));
        return false;
    }

    bool result = true;
    double r = rCmdParams["r"].toFloat(&result);
    if (!result)
    {
        cerr << qPrintable(QString(g_szSeparationFactorIncorrect));
        return false;
    }

    QString sTrainSet = rCmdParams["trainset"];
    QString sControlSet = rCmdParams["controlset"];
    int nTrainSamples, nTrainInputs, nTrainTargets, nControlSamples = 0;
    if (!load_trainset(sTrainSet, 0, 0, nTrainSamples, nTrainInputs,
                       nTrainTargets))
    {
        return false;
    }

    nControlSamples = round_bond005(r * nTrainSamples);
    if (nControlSamples <= 0)
    {
        cerr << qPrintable(QString(g_szSeparationFactorIncorrect));
        return false;
    }
    if (nControlSamples >= nTrainSamples)
    {
        cerr << qPrintable(QString(g_szSeparationFactorIncorrect));
        return false;
    }

    double *aTrainInputs = 0, *aTrainTargets = 0;
    double *aTempTrainSample = 0;
    int *aTrainIndexes = 0;
    try
    {
        aTrainInputs = new double[nTrainSamples * nTrainInputs];
        if (nTrainTargets > 0)
        {
            aTrainTargets = new double[nTrainSamples * nTrainTargets];
        }
        if (!load_trainset(sTrainSet, aTrainInputs, aTrainTargets,
                           nTrainSamples, nTrainInputs, nTrainTargets))
        {
            result = false;
        }
        else
        {
            size_t nInputDataSize = nTrainInputs * sizeof(double);
            size_t nTargetDataSize = nTrainTargets * sizeof(double);
            aTempTrainSample = new double[nTrainInputs + nTrainTargets];
            aTrainIndexes = new int[nTrainSamples];
            calculate_rand_indexes(aTrainIndexes, nTrainSamples);
            for (int i = 0; i < nTrainSamples; i++)
            {
                memcpy(&aTempTrainSample[0], &aTrainInputs[i * nTrainInputs],
                       nInputDataSize);
                memcpy(&aTrainInputs[i * nTrainInputs],
                       &aTrainInputs[aTrainIndexes[i] * nTrainInputs],
                       nInputDataSize);
                memcpy(&aTrainInputs[aTrainIndexes[i] * nTrainInputs],
                       &aTempTrainSample[0], nInputDataSize);

                if (nTrainTargets > 0)
                {
                    memcpy(&aTempTrainSample[0],
                           &aTrainTargets[i * nTrainTargets], nTargetDataSize);
                    memcpy(&aTrainTargets[i * nTrainTargets],
                           &aTrainTargets[aTrainIndexes[i] * nTrainTargets],
                           nTargetDataSize);
                    memcpy(&aTrainTargets[aTrainIndexes[i] * nTrainTargets],
                           &aTempTrainSample[0], nTargetDataSize);
                }
            }
            if (!save_trainset(sTrainSet, aTrainInputs, aTrainTargets,
                               nTrainSamples - nControlSamples, nTrainInputs,
                               nTrainTargets))
            {
                cerr << qPrintable(QString(g_szTrainsetWritingError).arg(
                                       sTrainSet));
                result = false;
            }
            else
            {
                double *aControlInputs
                        = &aTrainInputs[(nTrainSamples - nControlSamples)
                                        * nTrainInputs];
                double *aControlTargets = 0;
                if (nTrainTargets > 0)
                {
                    aControlTargets
                            = &aTrainTargets[(nTrainSamples - nControlSamples)
                                             * nTrainTargets];
                }
                if (!save_trainset(sControlSet, aControlInputs,aControlTargets,
                                   nControlSamples,nTrainInputs,nTrainTargets))
                {
                    cerr << qPrintable(QString(g_szControlsetWritingError).arg(
                                           sControlSet));
                    result = false;
                }
            }
        }
        if (aTrainInputs != 0)
        {
            delete[] aTrainInputs;
            aTrainInputs = 0;
        }
        if (aTrainTargets != 0)
        {
            delete[] aTrainTargets;
            aTrainTargets = 0;
        }
        if (aTrainIndexes != 0)
        {
            delete[] aTrainIndexes;
            aTrainIndexes = 0;
        }
        if (aTempTrainSample != 0)
        {
            delete[] aTempTrainSample;
            aTempTrainSample = 0;
        }
    }
    catch(...)
    {
        if (aTrainInputs != 0)
        {
            delete[] aTrainInputs;
            aTrainInputs = 0;
        }
        if (aTrainTargets != 0)
        {
            delete[] aTrainTargets;
            aTrainTargets = 0;
        }
        if (aTrainIndexes != 0)
        {
            delete[] aTrainIndexes;
            aTrainIndexes = 0;
        }
        if (aTempTrainSample != 0)
        {
            delete[] aTempTrainSample;
            aTempTrainSample = 0;
        }
        throw;
    }

    return result;
}

/* Преобразовать данные из формата CSV в формат обучающего множества.
   ВХОДНЫЕ АРГУМЕНТЫ
   rCmpParams - список параметров командной строки в виде пары "ключ-значение".
Для данной функции (преобразования данных из формата CSV в формат обучающего
множества) в списке должно быть пять ключей.
   1. "csv" - ключ, значением которого является название файла с данными
в формате CSV.
   2. "trainset" - ключ, значением которого является название файла с данными
в формате обучающего множества. Если такой файл существует, то он будет
перезаписан. Если же такого файла нет, то он будет создан.
   3. "i" - ключ, значением которого является размер входного сигнала
обучающего множества, формируемого на основе данных из CSV-файла.
   4. "o" - ключ, значением которого является размер выходного сигнала
обучающего множества, формируемого на основе данных из CSV-файла.
   5. "to_ts" - ключ, определяющий направление преобразования - to TrainSet,
т.е. из формата CSV в формат обучающего множества.
   ВОЗВРАЩАЕМЫЙ РЕЗУЛЬТАТ
   Если функция успешно завершила свою работу, то возвращается true. В случае
ошибки возвращается false и на экран выводится сообщение о соответствующей
ошибке. */
bool CSVtoTrainset(const TCmdParams& rCmdParams)
{
    if (!check_params_for_CSVtoTrainset(rCmdParams))
    {
        return false;
    }

    int nInputs = 0, nOutputs = 0, nSamples = 0;
    QString sTrainSet = rCmdParams["trainset"];
    QString sCSV = rCmdParams["csv"];
    bool result = true;

    nInputs = rCmdParams["i"].toInt(&result);
    if (result)
    {
        if (nInputs < 1)
        {
            result = false;
        }
    }
    if (!result)
    {
        cerr << qPrintable(QString(g_szIncorrectInputsNumber));
        return false;
    }

    nOutputs = rCmdParams["o"].toInt(&result);
    if (result)
    {
        if (nOutputs < 0)
        {
            result = false;
        }
    }
    if (!result)
    {
        cerr << qPrintable(QString(g_szIncorrectOutputsNumber));
        return false;
    }

    QList<QVector<double> > aCSVData;
    nSamples = readCSV(sCSV, nInputs + nOutputs, aCSVData);
    if (nSamples <= 0)
    {
        cerr << qPrintable(QString(g_szCSVReadingError).arg(sCSV));
        return false;
    }

    QFile trainsetFile(sTrainSet);
    if (!trainsetFile.open(QFile::WriteOnly | QFile::Truncate))
    {
        cerr << qPrintable(QString(g_szTrainsetWritingError).arg(sTrainSet));
        return false;
    }
    QDataStream trainsetStream(&trainsetFile);
    trainsetStream.setFloatingPointPrecision(QDataStream::DoublePrecision);
    if (trainsetStream.status() != QDataStream::Ok)
    {
        cerr << qPrintable(QString(g_szTrainsetWritingError).arg(sTrainSet));
        return false;
    }

    qint32 iTemp = nSamples;
    trainsetStream << iTemp;
    if (trainsetStream.status() != QDataStream::Ok)
    {
        cerr << qPrintable(QString(g_szTrainsetWritingError).arg(sTrainSet));
        return false;
    }

    iTemp = nInputs;
    trainsetStream << iTemp;
    if (trainsetStream.status() != QDataStream::Ok)
    {
        cerr << qPrintable(QString(g_szTrainsetWritingError).arg(sTrainSet));
        return false;
    }

    iTemp = nOutputs;
    trainsetStream << iTemp;
    if (trainsetStream.status() != QDataStream::Ok)
    {
        cerr << qPrintable(QString(g_szTrainsetWritingError).arg(sTrainSet));
        return false;
    }

    int iSample, i;
    QList<QVector<double> >::iterator it = aCSVData.begin();
    for (iSample = 0; iSample < nSamples; iSample++)
    {
        for (i = 0; i < nInputs; i++)
        {
            trainsetStream << (*it)[i];
        }
        if (trainsetStream.status() != QDataStream::Ok)
        {
            result = false;
            break;
        }
        for (i = 0; i < nOutputs; i++)
        {
            trainsetStream << (*it)[i+nInputs];
        }
        if (trainsetStream.status() != QDataStream::Ok)
        {
            result = false;
            break;
        }
        it++;
    }
    if (!result)
    {
        cerr << qPrintable(QString(g_szTrainsetWritingError).arg(sTrainSet));
        return false;
    }

    return true;
}

/* Преобразовать данные из формата обучающего множества в формат CSV.
   ВХОДНЫЕ АРГУМЕНТЫ
   rCmpParams - список параметров командной строки в виде пары "ключ-значение".
Для данной функции (преобразования данных из формата CSV в формат обучающего
множества) в списке должно быть три ключа.
   1. "csv" - ключ, значением которого является название файла с данными
в формате CSV. Если такой файл существует, то он будет перезаписан. Если же
такого файла нет, то он будет создан.
   2. "trainset" - ключ, значением которого является название файла с данными
в формате обучающего множества.
   3. "to_csv" - ключ, определяющий направление преобразования - to CSV, т.е. из
формата обучающего множества в формат CSV.
   ВОЗВРАЩАЕМЫЙ РЕЗУЛЬТАТ
   Если функция успешно завершила свою работу, то возвращается true. В случае
ошибки возвращается false и на экран выводится сообщение о соответствующей
ошибке. */
bool trainsetToCSV(const TCmdParams& rCmdParams)
{
    if (!check_params_for_trainsetToCSV(rCmdParams))
    {
        return false;
    }

    int nInputs = 0, nOutputs = 0, nSamples = 0;
    QString sTrainSet = rCmdParams["trainset"];
    QString sCSV = rCmdParams["csv"];
    bool result = true;
    double *aInputs = 0, *aOutputs = 0;

    QFile csvFile(sCSV);
    if (!csvFile.open(QFile::WriteOnly|QFile::Truncate|QFile::Text))
    {
        cerr << qPrintable(QString(g_szCSVReadingError).arg(sCSV));
        return false;
    }
    QTextStream csvStream(&csvFile);

    try
    {
        if (!load_trainset(sTrainSet, 0, 0, nSamples, nInputs, nOutputs))
        {
            cerr << qPrintable(QString(g_szTrainsetReadingError).arg(
                                   sTrainSet));
            result = false;
        }
        if (result)
        {
            aInputs = new double[nSamples * nInputs];
            if (nOutputs > 0)
            {
                aOutputs = new double[nSamples * nOutputs];
            }
            if (!load_trainset(sTrainSet, aInputs, aOutputs, nSamples, nInputs,
                               nOutputs))
            {
                cerr << qPrintable(QString(g_szTrainsetReadingError).arg(
                                       sTrainSet));
                result = false;
            }
        }
        if (result)
        {
            int iSample, i, iInput = 0, iOutput = 0;
            QString sReadLine, sSpace(", ");
            for (iSample = 0; iSample < nSamples; iSample++)
            {
                sReadLine = QString::number(aInputs[iInput]);
                for (i = 1; i < nInputs; i++)
                {
                    sReadLine += (sSpace + QString::number(aInputs[iInput+i]));
                }
                iInput += nInputs;
                if (nOutputs > 0)
                {
                    for (i = 0; i < nOutputs; i++)
                    {
                        sReadLine += (sSpace
                                      + QString::number(aOutputs[iOutput+i]));
                    }
                    iOutput += nOutputs;
                }
                csvStream << sReadLine << endl;
                if (csvStream.status() != QTextStream::Ok)
                {
                    result = false;
                    break;
                }
            }
            if (!result)
            {
                cerr << qPrintable(QString(g_szCSVWritingError).arg(sCSV));
            }
        }
        if (aInputs != 0)
        {
            delete[] aInputs;
            aInputs = 0;
        }
        if (aOutputs != 0)
        {
            delete[] aOutputs;
            aOutputs = 0;
        }
    }
    catch(...)
    {
        if (aInputs != 0)
        {
            delete[] aInputs;
            aInputs = 0;
        }
        if (aOutputs != 0)
        {
            delete[] aOutputs;
            aOutputs = 0;
        }
        throw;
    }

    return result;
}

/* Обработать "противоречивые" примеры обучающего множества, т.е. такие, у
которых входные сигналы одинаковы, а желаемые выходные сигналы разные. Под
обработкой понимается одна из трёх операций: вывод на экран номеров
противоречивых примеров, удаление всех противоречивых примеров либо же
объединение противоречивых примеров с одинаковыми входными сигналами.
   ВХОДНЫЕ АРГУМЕНТЫ
   rCmpParams - список параметров командной строки в виде пары "ключ-значение".
Для данной функции (разделения исходного обучающего множества на собственно
обучающее и контрольное подмножества) в списке должно быть два ключа:
"trainset" и "divergent":
   1. "trainset" - ключ, значением которого является строка с названием файла,
содержащего анализируемое обучающее множество.
   2. "divergent" - ключ, значением которого является команда обработки
противоречивых примеров обучающего множества: "show", "remove" или "unite".
   ВОЗВРАЩАЕМЫЙ РЕЗУЛЬТАТ
   Если функция успешно завершила свою работу, то возвращается true. В случае
ошибки возвращается false и на экран выводится сообщение о соответствующей
ошибке. */
bool processDivergentTrainSamples(const TCmdParams& rCmdParams)
{
    if (!check_params_for_processDivergentTrainSamples(rCmdParams))
    {
        return false;
    }

    QString sTrainSet = rCmdParams["trainset"];
    QString sCommandName = rCmdParams["divergent"];
    int nCommandType, nSamples = 0, nInputs = 0, nOutputs = 0;
    double *aInputs = 0, *aOutputs = 0;
    bool result = true;

    if (sCommandName.compare(g_szRemoveDivergentSamples,
                             Qt::CaseInsensitive) == 0)
    {
        nCommandType = 1;
    }
    else if (sCommandName.compare(g_szUniteDivergentSamples,
                                  Qt::CaseInsensitive) == 0)
    {
        nCommandType = 2;
    }
    else
    {
        nCommandType = 0;
    }

    try
    {
        if (load_trainset(sTrainSet, 0, 0, nSamples, nInputs, nOutputs))
        {
            if (nOutputs <= 0)
            {
                result = false;
                cerr << qPrintable(g_szDivergentSearchImpossible);
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szTrainsetReadingError).arg(
                                   sTrainSet));
            result = false;
        }
        if (result)
        {
            aInputs = new double[nSamples * nInputs];
            aOutputs = new double[nSamples * nOutputs];
            if (!load_trainset(sTrainSet, aInputs, aOutputs, nSamples, nInputs,
                               nOutputs))
            {
                cerr << qPrintable(QString(g_szTrainsetReadingError).arg(
                                       sTrainSet));
                result = false;
            }
        }
        if (result)
        {
            QList<QList<int> > aGroupsOfDivergentSamples;
            find_divergent_samples_in_train_set(
                        aInputs, aOutputs, nSamples, nInputs, nOutputs,
                        aGroupsOfDivergentSamples);
            if (aGroupsOfDivergentSamples.isEmpty())
            {
                cout << qPrintable(g_szNoDivergentSamples) << std::endl;
            }
            else if ((aGroupsOfDivergentSamples.size() == 1)
                     && (aGroupsOfDivergentSamples.at(0).size() == nSamples))
            {
                cout << qPrintable(g_szAllSamplesAreDivergent) << std::endl;
            }
            else
            {
                if (nCommandType == 0)
                {
                    print_divergent_samples(aGroupsOfDivergentSamples);
                }
                else
                {
                    if (nCommandType == 1)
                    {
                        nSamples = remove_divergent_samples(
                                    aInputs,aOutputs,nSamples,nInputs,nOutputs,
                                    aGroupsOfDivergentSamples);
                    }
                    else
                    {
                        TSolvedTask task;
                        if (rCmdParams["task"].compare(g_szClassificationTask,
                                                       Qt::CaseInsensitive)==0)
                        {
                            task = taskCLASSIFICATION;
                        }
                        else
                        {
                            task = taskREGRESSION;
                        }
                        nSamples = unite_divergent_samples(
                                    aInputs,aOutputs,nSamples,nInputs,nOutputs,
                                    aGroupsOfDivergentSamples, task);
                    }
                    if (!save_trainset(sTrainSet, aInputs, aOutputs,
                                       nSamples, nInputs, nOutputs))
                    {
                        cerr<<qPrintable(QString(g_szTrainsetWritingError).arg(
                                             sTrainSet));
                        result = false;
                    }
                }
            }
        }
        if (aInputs != 0)
        {
            delete[] aInputs;
            aInputs = 0;
        }
        if (aOutputs != 0)
        {
            delete[] aOutputs;
            aOutputs = 0;
        }
    }
    catch(...)
    {
        if (aInputs != 0)
        {
            delete[] aInputs;
            aInputs = 0;
        }
        if (aOutputs != 0)
        {
            delete[] aOutputs;
            aOutputs = 0;
        }
        throw;
    }

    return result;
}

/* Удалить все повторяющиейся примеры обучающего множества, т.е. такие, у
которых одинаковы входные и желаемые выходные сигналы. В каждой группе
повторяющихся примеров обучающего множества после удаления остаётся только один
пример.
   ВХОДНЫЕ АРГУМЕНТЫ
   rCmpParams - список параметров командной строки в виде пары "ключ-значение".
Для данной функции (разделения исходного обучающего множества на собственно
обучающее и контрольное подмножества) в списке должно быть два ключа:
"trainset" и "divergent":
   1. "trainset" - ключ, значением которого является строка с названием файла,
содержащего анализируемое обучающее множество.
   2. "repeat" - ключ без значения указывающий необходимость удаления
повторяющихся примеров обучающего множества.
   ВОЗВРАЩАЕМЫЙ РЕЗУЛЬТАТ
   Если функция успешно завершила свою работу, то возвращается true. В случае
ошибки возвращается false и на экран выводится сообщение о соответствующей
ошибке. */
bool deleteRepeatingTrainSamples(const TCmdParams& rCmdParams)
{
    if (!check_params_for_deleteRepeatingTrainSamples(rCmdParams))
    {
        return false;
    }

    QString sTrainSet = rCmdParams["trainset"];
    int nSamples = 0, nInputs = 0, nOutputs = 0;
    double *aInputs = 0, *aOutputs = 0;
    bool result = true;

    try
    {
        if (!load_trainset(sTrainSet, 0, 0, nSamples, nInputs, nOutputs))
        {
            cerr << qPrintable(QString(g_szTrainsetReadingError).arg(
                                   sTrainSet));
            result = false;
        }
        if (result)
        {
            aInputs = new double[nSamples * nInputs];
            if (nOutputs > 0)
            {
                aOutputs = new double[nSamples * nOutputs];
            }
            if (!load_trainset(sTrainSet, aInputs, aOutputs, nSamples, nInputs,
                               nOutputs))
            {
                cerr << qPrintable(QString(g_szTrainsetReadingError).arg(
                                       sTrainSet));
                result = false;
            }
        }
        if (result)
        {
            QList<QList<int> > aGroupsOfRepeatingSamples;
            find_repeating_samples_in_train_set(
                        aInputs, aOutputs, nSamples, nInputs, nOutputs,
                        aGroupsOfRepeatingSamples);
            if (!aGroupsOfRepeatingSamples.isEmpty())
            {
                nSamples = remove_repeating_samples(
                            aInputs, aOutputs, nSamples, nInputs, nOutputs,
                            aGroupsOfRepeatingSamples);
                if (!save_trainset(sTrainSet, aInputs, aOutputs,
                                   nSamples, nInputs, nOutputs))
                {
                    cerr << qPrintable(QString(g_szTrainsetWritingError).arg(
                                           sTrainSet));
                    result = false;
                }
            }
        }
        if (aInputs != 0)
        {
            delete[] aInputs;
            aInputs = 0;
        }
        if (aOutputs != 0)
        {
            delete[] aOutputs;
            aOutputs = 0;
        }
    }
    catch(...)
    {
        if (aInputs != 0)
        {
            delete[] aInputs;
            aInputs = 0;
        }
        if (aOutputs != 0)
        {
            delete[] aOutputs;
            aOutputs = 0;
        }
        throw;
    }

    return result;
}
