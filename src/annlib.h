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

#ifndef ANNLIB_H
#define ANNLIB_H

#include <cmath>
#include <exception>

#include <QObject>

#include "mathlib_bond005.h"
#include "randlib.h"

typedef enum {taskCLASSIFICATION, taskREGRESSION} TSolvedTask;

/*****************************************************************************/
/* ОБЪЯВЛЕНИЯ ФУНКЦИЙ */
/*****************************************************************************/

/* Выполнить SOFTMAX-нормализацию вектора данных data[] длиной n. */
void do_softmax_normalization(double data[], int n);

/* Перемешивание элементов в некотором массиве длиной nLength путём генерации
случайных индексов этих элементов aIndexes. */
void calculate_rand_indexes(int aIndexes[], int nLength);

/* Функция возвращает случайное целое число из равномерного
распределения в диапазоне [min_value; max_value]. Используется системная
функция rand(). */
inline int get_random_value(int min_value, int max_value)
{
    double temp = (generate_random_value() * (max_value - min_value)
                   + min_value);
    return round_bond005(temp);
}

/* Вычислить среднеабсолютную ошибку регрессии */
double regression_error(double output, double target);

/* Сравнить по содержимому два сигнала обучающего множества aSignal1[] и
aSignal2[] (неважно, входные это сигналы или желаемые выходные) одинакового
размера nSignalSize. Вернуть true, если все компоненты входных сигналов
одинаковы, и false, если обнаружены различия */
bool same_train_signals(const double aSignal1[], const double aSignal2[],
                        int nSignalSize);

/* Найти номер максимального компонента в сигнале aSignal[] размером
nSignalSize. */
int find_maximum_component(const double aSignal[], int nSignalSize);

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
bool load_trainset(const QString& sFileName, double aTrainInputs[],
                   double aTrainTargets[], int& nTrainSamples,
                   int& nTrainInputs, int& nTrainTargets);

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
bool save_trainset(const QString& sFileName, const double aTrainInputs[],
                   const double aTrainTargets[], int nTrainSamples,
                   int nTrainInputs, int nTrainTargets);

/*****************************************************************************/
/* ОБЪЯВЛЕНИЯ КЛАССОВ */
/*****************************************************************************/

// Родительский класс исключений для данной нейросетевой библиотеки
class EANNError: public std::exception
{
protected:
    QString m_sErrorMessage;
public:
    ~EANNError() throw() {}
    const char* what() const throw();
};


/* Класс исключений, генерируемых в случае, когда размеры нейросети (размер
входного сигнала, либо количество слоёв, либо размер одного из слоёв) заданы
некорректно.
   Все размеры нейросети - это положительные числа.
   Потомок класса EANNError. Ни для кого не является родительским классом. */
class EMLPStructError: public EANNError
{
public:
    EMLPStructError() throw();
};

/* Класс исключений, генерируемых в случае, когда количество примеров
обучающего множества или множества тестовых данных задано некорректно.
   Некорректным является отрицательное либо нулевое число примеров.
   Потомок класса EANNError. Ни для кого не является родительским классом. */
class ETrainSetError: public EANNError
{
public:
    ETrainSetError() throw();
    ETrainSetError(const QString& sErrorMsg) throw();
};


/* Класс исключений, связанных с ошибками алгоритма обучения, т.е. когда
мы пытались установить некорректные значения параметров алгоритма обучения
(напр., неположительное максимальное число итераций, неположительный
коэффициент скорости обучения и т.п.).
   Потомок класса EANNError. Родительский класс для исключений. Ни для кого
не является родительским классом. */
class ETrainProcessError: public EANNError
{
public:
    ETrainProcessError(const QString& sIncorrectParamName) throw();
};


typedef enum {LIN, SIG, SOFT} TActivationKind;

// Класс для реализации многослойного персептрона
class CMultilayerPerceptron
{
public:
    CMultilayerPerceptron();
    CMultilayerPerceptron(int nInputs, int nLayers, const int aLayerSizes[],
                          const TActivationKind aActivations[]);
    CMultilayerPerceptron(const CMultilayerPerceptron& src);
    CMultilayerPerceptron& operator= (const CMultilayerPerceptron& src);
    ~CMultilayerPerceptron();

    /* Вычисление последовательности выходных сигналов inputs[] многослойного
    персептрона при подаче на вход соответствующей последовательности входных
    сигналов outputs[].
       Длина последовательности входных сигналов и соответствующей
    последовательности вычисляемых выходных сигналов равна nSamples. */
    void calculate_outputs(const double inputs[], double outputs[],
                           int nSamples);

    /* Вычисление среднеквадратичного отклонения между последовательностью
    желаемых выходных сигналов targets[] и последовательностью реальных
    выходных сигналов многослойного персептрона, вычисленных при подаче
    на вход соответствующей последовательности входных сигналов inputs[].
       Длина последовательности входных сигналов и соответствующей
    последовательности желаемых выходных сигналов равна nSamples.
       Возвращаемое значение - вычисленное среднеквадратичное отклонение. */
    double calculate_mse(const double inputs[], const double targets[],
                         int nSamples);

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
    double calculate_mse(const double inputs[], const double targets[],
                         const double distribution[], int nSamples);

    /* Вычисление ошибки классификации или регресии в процентах на
    последовательности входных сигналов inputs[]. Желаемые (эталонные) выходные
    сигналы заданы массивом targets[].
       Длина последовательности входных сигналов и соответствующей
    последовательности желаемых выходных сигналов равна nSamples.
       Возвращаемое значение - вычисленная ошибка в процентах (от 0 до 100). */
    double calculate_error(const double inputs[], const double targets[],
                           int nSamples, TSolvedTask task);

    /* Вычисление ошибки классификации или регресии в процентах на
    последовательности входных сигналов inputs[]. Желаемые (эталонные) выходные
    сигналы заданы массивом targets[].
       Длина последовательности входных сигналов и соответствующей
    последовательности желаемых выходных сигналов равна nSamples.
       Распределение вероятностей последовательности входных сигналов inputs[]
    задано массивом distribution[] (длина массива равна nSamples - по числу
    примеров в тестовом множестве).
       Возвращаемое значение - вычисленная ошибка в процентах (от 0 до 100). */
    double calculate_error(const double inputs[], const double targets[],
                           const double distribution[], int nSamples,
                           TSolvedTask task);

    // Инициализировать весовые коэффициенты сети случайными значениями.
    void initialize_weights();

    // Загрузка параметров многослойного персептрона из файла
    bool load(const QString& sFilename);

    // Сохранение параметров многослойного персептрона в файл
    bool save(const QString& sFilename) const;

    /* Изменение размеров нейросети (числа входов, числа слоёв, числа
       нейронов в каждом из слоёв) */
    void resize(int nInputs, int nLayers, const int aLayerSizes[],
                const TActivationKind aActivations[]);

    // Метод доступа к свойству "КОЛИЧЕСТВО ВХОДОВ"
    inline int getInputsCount() const
        { return m_nInputsCount; }

    // Метод доступа к свойству "КОЛИЧЕСТВО СЛОЁВ"
    inline int getLayersCount() const
        { return m_nLayersCount; }

    // Метод доступа к свойству "РАЗМЕРЫ СЛОЁВ"
    inline int getLayerSize(int iLayerNo) const
        { return m_aLayerSizes[iLayerNo]; }

    // Методы доступа к свойству "АКТИВАЦИОННЫЕ ФУНКЦИИ НЕЙРОНОВ В СЛОЯХ"
    inline TActivationKind getActivationKind(int iLayerNo) const
    {
        return m_aActivations[iLayerNo];
    }
    inline void setActivationKind(int iLayerNo, TActivationKind kind)
    {
        m_aActivations[iLayerNo] = kind;
    }

    /* Метод доступа к свойству "РАЗМЕРЫ ВХОДНЫХ СИГНАЛОВ ДЛЯ СЛОЁВ"
       (только для чтения) */
    inline int getInputsCountOfLayer(int iLayerNo) const
    {
        return m_aInputsCount[iLayerNo];
    }

    // Методы доступа к свойству "ВЕСОВЫЕ КОЭФФИЦИЕНТЫ СЕТИ"
    inline double getWeight(int iLayerNo, int iNeuronNo, int iWeightNo) const
    {
        return m_aWeights[m_aIndexesForIDBD[iLayerNo]
                          + iNeuronNo * (m_aInputsCount[iLayerNo] + 1)
                          + iWeightNo];
    }
    inline void setWeight(int iLayerNo, int iNeuronNo, int iWeightNo,
                          double weight)
    {
        m_aWeights[m_aIndexesForIDBD[iLayerNo] + iNeuronNo
                   * (m_aInputsCount[iLayerNo] + 1) + iWeightNo] = weight;
    }

    /* Метод доступа к свойству "ОБЩЕЕ КОЛИЧЕСТВО ВЕСОВ СЕТИ"
       (только для чтения) */
    inline int getAllWeightsCount() const
    {
        return (m_aIndexesForIDBD[m_nLayersCount - 1]
                + m_aLayerSizes[m_nLayersCount - 1]
                * (m_aInputsCount[m_nLayersCount - 1] + 1));
    }
private:
    int m_nInputsCount;  // свойство "КОЛИЧЕСТВО ВХОДОВ"
    int m_nLayersCount;  // свойство "КОЛИЧЕСТВО СЛОЁВ"
    int *m_aLayerSizes;  // свойство "РАЗМЕРЫ СЛОЁВ" (массив)
    int *m_aInputsCount; // свойство "РАЗМЕРЫ ВХОДНЫХ СИГНАЛОВ ДЛЯ СЛОЁВ"
    double *m_aWeights;  // свойство "ВЕСОВЫЕ КОЭФФИЦИЕНТЫ" (3-мерный массив)
    TActivationKind *m_aActivations; /*свойство "АКТИВАЦИОННЫЕ ФУНКЦИИ нейронов
                                       в слоях" (массив: для каждого слоя -
                                       свой тип активационной функции) */

    int *m_aIndexesForIDBD; /* для каждого слоя - начальный индекс подмассива
                               его весовых коэффициентов в общем массиве
                               m_aWeights */
    double *m_aTempOutputs;/* промежуточный массив для хранения выходов
                              текущего слоя */
    double *m_aTempInputs; /* промежуточный массив для хранения выходов
                              предыдущего слоя (т.е. входов в текущий слой) */

    /* Копирование значений атрибутов другого многослойного персептрона
    (private-операция, которая используется в копирующем присваивании и
    в операции загрузки из файла). */
    void copy_from(const CMultilayerPerceptron& src);

    /* Проверка корректности заданных размеров нейросети: количества входов
    nInputs, количества слоёв nLayers и количества нейронов в слоях
    aLayerSizes[]. В случае, если значение хотя бы одного из размеров
    некорректно, генерируется исключение EMLPSizeError.
       Данная операция является private-операций, используемой в конструкторе
    и операции resize для проверки, правильно ли заданы размеры нейросети. */
    void check_size(int nInputs, int nLayers, const int aLayerSizes[]);
};

enum TTrainingState {tsCONTINUE, tsMAXEPOCHS, tsUSER, tsGRADIENT, tsMINIMUM};

class CTrainingOfMLP: public QObject
{
    Q_OBJECT
private:
    double *m_aWeightsOfTrainSamples;/*распределение весов примеров в обучающем
                                       множестве (сумма весов всех примеров
                                       равна количеству примеров обучающего
                                       множества). */
    int m_nMaxEpochsCount; /* максимально допустимое число эпох обучения (по
                              умолчанию 100). */
    volatile TTrainingState m_state; /* состояние процесса обучения */

    double *m_aTrainInputs; /* входные сигналы обучающего множества */
    double *m_aTrainTargets;/* желаемые выходные сигналы обучающего множества*/
    int m_nTrainSamples;    /* количество обучающих примеров (пар "входной
                               сигнал - желаемый выходной сигнал") */
    int *m_aIndexesOfTrainInputs; /* индексы начальных компонент входных
                                     сигналов обучающего множества */
    int *m_aIndexesOfTrainTargets;/* индексы начальных компонент желаемых
                                     выходных сигналов обучающего множества */

    double *m_aNetOutputs; /* выходы нейронов обучаемой нейросети */
    double *m_aNetOutputsD;/* производные выходов нейронов обучаемой
                              нейросети */
    int *m_aNetOutputsI; /* массив начальных индексов выходов каждого слоя */

    void getmem_for_net_outputs();
    void clear_data();
public:
    CTrainingOfMLP(QObject* pobj = 0);
    virtual ~CTrainingOfMLP() {}

    /* Запустить процесс обучения нейронной сети pTrainedMLP на обучающем
       множестве pTrainSet. */
    void train(CMultilayerPerceptron *pTrainedMLP, double aTrainInputs[],
               double aTrainTargets[], int nTrainSamples);

    /* Запустить процесс обучения нейронной сети pTrainedMLP на обучающем
       множестве pTrainSet. Распределение вероятности примеров задано
       массивом sDistribution[]. */
    void train(CMultilayerPerceptron *pTrainedMLP, double aTrainInputs[],
               double aTrainTargets[], double aDistribution[],int nTrainSamples);

    // Методы доступа к свойству "МАКСИМАЛЬНОЕ ЧИСЛО ЭПОХ"
    inline int getMaxEpochsCount() const { return m_nMaxEpochsCount; }
    void setMaxEpochsCount(int nMaxEpochsCount);
protected:
    CMultilayerPerceptron *m_pTrainedMLP;

    /* Функция возвращает размер обучающего множества (количество обучающих
    примеров в нём) */
    inline int getNumberOfTrainSamples() { return m_nTrainSamples; }

    /* Функция возвращает iInput-й компонент входного сигнала в iSample-ом
    обучающем примере. */
    inline double getTrainInput(int iSample, int iInput)
    {
        return m_aTrainInputs[m_aIndexesOfTrainInputs[iSample] + iInput];
    }

    /* Функция возвращает iTarget-й компонент желаемого выходного сигнала
    в iSample-ом обучающем примере. */
    inline double getTrainTarget(int iSample, int iTarget)
    {
        return m_aTrainTargets[m_aIndexesOfTrainTargets[iSample] + iTarget];
    }

    /* Функция возвращает вес iSample-го примера обучающего множества*/
    inline double getSampleWeight(int iSample) const
    {
        return m_aWeightsOfTrainSamples[iSample];
    }

    /* Функция возвращает значение выхода iNeuron-го нейрона iLayer-го слоя
    обучаемой нейросети после подачи на её входы очередного примера обучающего
    множества (с помощью функции calculate_outputs). */
    inline double getNetOutput(int iLayer, int iNeuron) const
    {
        return m_aNetOutputs[m_aNetOutputsI[iLayer] + iNeuron];
    }
    /* Функция возвращает значение производной выхода iNeuron-го нейрона
    iLayer-го слоя обучаемой нейросети после подачи на её входы очередного
    примера обучающего множества (с помощью функции calculate_outputs). */
    inline double getNetOutputD(int iLayer, int iNeuron) const
    {
        return m_aNetOutputsD[m_aNetOutputsI[iLayer] + iNeuron];
    }

    /* Подать на вход нейросети pNet входной сигнал из текущего примера
    обучающего множества и вычислить выходы и производные выходов всех нейронов
    этой нейросети.
       По умолчанию указатель на нейросеть pNet является нулевым. В этом случае
    вычисляются выходы и производные нейронов обучаемой нейросети
    m_pTrainedNet. */
    void calculate_outputs_and_derivatives(int iSample,
                                           CMultilayerPerceptron* pNet = 0);

    /* Подать на вход нейросети pNet входной сигнал из текущего примера
    обучающего множества и вычислить выходы всех нейронов этой нейросети.
       По умолчанию указатель на нейросеть pNet является нулевым. В этом случае
    вычисляются выходы нейронов обучаемой нейросети m_pTrainedNet. */
    void calculate_outputs(int iSample, CMultilayerPerceptron* pNet = 0);

    virtual void initialize_training() {}
    virtual void finalize_training() {}
    virtual TTrainingState do_epoch(int nEpoch) = 0;
signals:
    void start_training();
    void do_training_epoch(int nEpochsCount);
    void end_training(TTrainingState state);
public slots:
    void stop_training_state()
    {
        m_state = tsUSER;
    }
};

/* Класс для реализации алгоритма обучения по методу стохастического обратного
распространения ошибки (корректировка весов происходит после предъявления
каждого обучающего примера, а не после анализа всех примеров обучающего
множества, как это реализуется в пакетном обратном распространении). Алгоритм
может работать в двух вариантах:
   1) классический backprop, когда коэффициент скорости обучения одинаков
для всех весов сети и не меняется в процессе обучения;
   2) Incremental Delta Bar Delta (IDBD), когда коэффициент скорости обучения
автоматически подбирается для каждого настраиваемого веса и изменяется в
процессе обучения.
   Для реализации первого варианта необходимо установить свойство
"АДАПТИВНОСТЬ СКОРОСТИ ОБУЧЕНИЯ" в false, а для второго варианта - в true.
   Во втором случае свойство "КОЭФФИЦИЕНТ СКОРОСТИ ОБУЧЕНИЯ" содержит начальное
приближения коэффициента скорости обучения, которое затем будет меняться в
соответствии с алгоритмом IDBD. */
class COnlineBackpropTraining: public CTrainingOfMLP
{
    Q_OBJECT
public:
    COnlineBackpropTraining(QObject* pobj = 0);
    ~COnlineBackpropTraining();
    //virtual ~COnlineBackpropTraining();

    // Методы доступа к свойству "НАЧАЛЬНЫЙ КОЭФФИЦИЕНТ СКОРОСТИ ОБУЧЕНИЯ"
    inline double getStartLearningRateParam() { return m_startRate; }
    void setStartLearningRateParam(double value);

    // Методы доступа к свойству "КОНЕЧНЫЙ КОЭФФИЦИЕНТ СКОРОСТИ ОБУЧЕНИЯ"
    inline double getFinalLearningRateParam() { return m_finalRate; }
    void setFinalLearningRateParam(double value);

    // Методы доступа к свойству "ПАРАМЕТР АДАПТАЦИИ СКОРОСТИ ОБУЧЕНИЯ"
    inline double getTheta() { return m_theta; }
    void setTheta(double value);

    // Методы доступа к свойству "АДАПТИВНОСТЬ СКОРОСТИ ОБУЧЕНИЯ"
    inline bool getAdaptiveRate() { return m_bAdaptiveRate; }
    void setAdaptiveRate(bool value);
private:
    double m_rate; // свойство "КОЭФФИЦИЕНТ СКОРОСТИ ОБУЧЕНИЯ"
    double m_startRate;// свойство "НАЧАЛЬНЫЙ КОЭФФИЦИЕНТ СКОРОСТИ ОБУЧЕНИЯ"
    double m_finalRate;// свойство "КОНЕЧНЫЙ КОЭФФИЦИЕНТ СКОРОСТИ ОБУЧЕНИЯ"
    double m_theta; // свойство "ПАРАМЕТР АДАПТАЦИИ СКОРОСТИ ОБУЧЕНИЯ"
    bool m_bAdaptiveRate;  // свойство "АДАПТИВНОСТЬ СКОРОСТИ ОБУЧЕНИЯ"

    int *m_aIndexesOfTrainSamples; /* массив случайно перемешанных индексов
                                      обучающих примеров */

    double *m_aLocalGradients1, *m_aLocalGradients2;

    int *m_aIndexesForIDBD; /* для каждого слоя - начальный индекс подмассивов
                               его коэффициентов "Betta" и "H" в общих массивах
                               m_aBetta и m_aH соответственно */
    double *m_aBetta, *m_aH;/* Свойства "Betta" и "H", используемые в
                               алгоритме адаптации скорости обучения */

    /* Выполнить корректировку весов сети по классическому алгоритму Online
       Backprop при условии, что на сеть распространено входное воздействие
       текущего примера обучающего множества. */
    void change_weights_by_BP(int iSample);

    /* Выполнить корректировку весов сети по алгоритму Incremental Delta Bar
       Delta при условии, что на сеть распространено входное воздействие
       текущего примера обучающего множества. */
    void change_weights_by_IDBD(int iSample);

    /* Методы доступа к свойству "Betta" (данное свойство используется при
    обучении с адаптацией скорости) */
    inline double getBetta(int iLayerNo, int iNeuronNo, int iWeightNo)
        {
            return m_aBetta[m_aIndexesForIDBD[iLayerNo] + iNeuronNo
                            * (m_pTrainedMLP->getInputsCountOfLayer(iLayerNo)
                               + 1)
                            + iWeightNo];
        }
    inline void setBetta(int iLayerNo, int iNeuronNo, int iWeightNo,
                         double value)
        {
            m_aBetta[m_aIndexesForIDBD[iLayerNo] + iNeuronNo
                     * (m_pTrainedMLP->getInputsCountOfLayer(iLayerNo) + 1)
                     + iWeightNo] = value;
        }

    /* Методы доступа к свойству "H" (данное свойство используется при
    обучении с адаптацией скорости) */
    inline double getH(int iLayerNo, int iNeuronNo, int iWeightNo)
        {
            return m_aH[m_aIndexesForIDBD[iLayerNo] + iNeuronNo
                        * (m_pTrainedMLP->getInputsCountOfLayer(iLayerNo) + 1)
                        + iWeightNo];
        }
    inline void setH(int iLayerNo, int iNeuronNo, int iWeightNo, double value)
        {
            m_aH[m_aIndexesForIDBD[iLayerNo] + iNeuronNo
                 * (m_pTrainedMLP->getInputsCountOfLayer(iLayerNo) + 1)
                 + iWeightNo] = value;
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
    void initialize_Betta_and_H();
protected:
    // Выделить память для всех промежуточных переменных и инициализировать их
    void initialize_training();

    // Освободить память ото всех промежуточных переменных
    void finalize_training();

    // Выполнить очередную эпоху обучения
    TTrainingState do_epoch(int nEpoch);
};

/* Абстрактрный класс для реализации пакетных алгоритмов обучения на основе
обратного распространения ошибки (корректировка весов происходит только после
анализа всех примеров обучающего множества, а не после предъявления каждого
обучающего примера, как это реализуется в стохастическом обратном
распространении).
   Поскольку над алгоритмом пакетного обратного распространения надстроено
множество алгоритмов обучения, таких как Resilient Backprop, метод сопряжённых
градиентов и т.п., то функция change_weights(), предназначенная для обновления
весов сети и автоматически вызываемая после пересчёта градиента по этим весам,
описана как абстрактная.
   Доступ к суммарному градиенту по весам сети, вычисляемому на основе всех
примеров обучающего множества, осуществляется с помощью свойства "MeanG"
(protected-функции getMeanG и SetMeanG). */
class CBatchBackpropTraining: public CTrainingOfMLP
{
    Q_OBJECT
public:
    CBatchBackpropTraining(QObject* pobj = 0);
    ~CBatchBackpropTraining();

    inline double getEpsilon() { return m_epsilon; }
    void setEpsilon(double value);
private:
    double *m_aMeanG;/* свойство "ВЕКТОР СУММАРНОГО ГРАДИЕНТА" (вектор
                        суммарного градиента вычисляется на основе всех
                        примеров обучающего множества; компонент вектора
                        насчитывается столько, сколько синаптических
                        весов и смещений сети) */
    double *m_aCurG; /* свойство "ВЕКТОР ТЕКУЩЕГО ГРАДИЕНТА" (вектор
                        текущего градиента вычисляется на основе одного,
                        текущего, примера обучающего множества; компонент
                        вектора насчитывается столько, сколько
                       синаптических весов и смещений сети) */

    double *m_aLocalGradients1, *m_aLocalGradients2;
    int *m_aGradientIndexes; /* для каждого слоя - начальный индекс подмассивов
                                его компонент вектора градиента в общем массиве
                                m_aMeanG и m_aCurG */

    double m_initGradientNorm;/*евклидова норма вектора суммарного градиента
                                m_aMeanG, вычисленная перед первой эпохой */
    double m_meanGradientNorm;/*евклидова норма вектора суммарного градиента
                                m_aMeanG */
    double m_meanError;      /* средняя ошибка нейросети на обучающем
                                множестве */
    double m_epsilon;/* критерий останова обучения: если выполняется нер-во
                        m_meanGradientNorm < m_epsilon * m_initGradientNorm,
                        то обучение завершается. */

    /* Вычислить вектор текущего градиента по всем весам и смещениям обучаемой
       сети на основе текущего примера обучающего множества. */
    void calculate_cur_gradient(int iSample);

    /* Методы доступа к свойству "CurG" (вектор текущего градиента) */
    inline double getCurGradient(int iLayerNo, int iNeuronNo, int iWeightNo)
    {
        return m_aCurG[m_aGradientIndexes[iLayerNo] + iNeuronNo
                       * (m_pTrainedMLP->getInputsCountOfLayer(iLayerNo) + 1)
                       + iWeightNo];
    }
    inline void setCurGradient(int iLayerNo, int iNeuronNo, int iWeightNo,
                               double value)
    {
        m_aCurG[m_aGradientIndexes[iLayerNo] + iNeuronNo
                * (m_pTrainedMLP->getInputsCountOfLayer(iLayerNo) + 1)
                + iWeightNo] = value;
    }
protected:
    void initialize_training();
    void finalize_training();
    TTrainingState do_epoch(int nEpoch);

    /* Выполнить корректировку весов сети. */
    virtual void change_weights(int nEpoch, TTrainingState& training_state)=0;

    /* Функция возвращает элемент вектора суммарного градиента. */
    inline double getMeanGradient(int iLayerNo, int iNeuronNo, int iWeightNo)
    {
        return m_aMeanG[m_aGradientIndexes[iLayerNo] + iNeuronNo
                        * (m_pTrainedMLP->getInputsCountOfLayer(iLayerNo) + 1)
                        + iWeightNo];
    }

    // Метод возвращает значение эвклидовой нормы вектора суммарного градиента
    inline double getMeanGradientNorm() { return m_meanGradientNorm; }

    //Метод возвращает значение средней ошибки нейросети на обучающем множестве
    inline double getMeanError() { return m_meanError; }
};

/* Класс для реализации алгоритма обучения по методу "упругого" обратного
распространения ошибки RPROP (Resilient Backprop).
   RPROP является надстройкой над пакетным обратным распространением. */
class CResilientBackpropTraining: public CBatchBackpropTraining
{
    Q_OBJECT
public:
    CResilientBackpropTraining(QObject* pobj = 0);
    ~CResilientBackpropTraining();

    // Методы доступа к свойству "НАЧАЛЬНОЕ ЗНАЧЕНИЕ СКОРОСТИ ОБУЧЕНИЯ"
    inline double getInitialLearningRate() { return m_initLearningRate; }
    void setInitialLearningRate(double learning_rate);

    // Получение минимально допустимого значения скорости обучения
    inline double getMinLearningRate() { return m_minLearningRate; }

    // Получение максимально допустимого значения скорости обучения
    inline double getMaxLearningRate() { return m_maxLearningRate; }
private:
    double m_minLearningRate;
    double m_maxLearningRate;
    double m_initLearningRate;// свойство "НАЧАЛЬНОЕ ЗНАЧЕНИЕ СКОРОСТИ ОБУЧЕНИЯ"
    double *m_aRates;/* свойство "ТЕКУЩИЕ КОЭФФИЦИЕНТЫ СКОРОСТИ ОБУЧЕНИЯ"
                        (индивидуальные для каждого синаптического веса и
                        смещения обучаемой сети) */
    double *m_aPrevG;// предыдущий вектор суммарного градиента обучаемой сети
protected:
    void change_weights(int nEpoch, TTrainingState& training_state);

    // Выделить память для всех промежуточных переменных и инициализировать их
    void initialize_training();

    // Освободить память ото всех промежуточных переменных
    void finalize_training();
};

/* Класс для реализации алгоритма обучения по наискорейшего спуска (значение
свойства m_bConjugateGradient устанавливается в false) либо по методу
сопряжённых градиентов (значение свойства m_bConjugateGradient устанавливается
в true). Оптимальная длина шага в выбранном направлении находится в диапазоне
[m_minLearningRate; m_maxLearningRate] и находится по методу Брента. В случае,
если задан метод сопряжённых градиентов, то множитель масштабирования Betta
вычисляется по формуле Полака-Рибьера. Рестарт метода сопрядённых градиентов
происходит, если множитель масштабирования принимает неположительные значения.
   Алгоритм найскорейшего спуска и алгоритм сопряжённых градиентов являются
надстройкой над пакетным обратным распространением. */
class CGradientDescentTraining: public CBatchBackpropTraining
{
    Q_OBJECT
public:
    CGradientDescentTraining(QObject* pobj = 0);
    ~CGradientDescentTraining();

    /* Методы доступа к свойству "МАКСИМАЛЬНОЕ ЗНАЧЕНИЕ СКОРОСТИ ОБУЧЕНИЯ". */
    inline double getMaxLearningRate() { return m_maxLearningRate; }
    void setMaxLearningRate(double value);

    inline bool isConjugateGradient() { return m_bConjugateGradient; }
    inline void setConjugateGradient(bool value)
    {
        m_bConjugateGradient = value;
    }

    /* Методы доступа к свойству "МАКС.ЧИСЛО ИТЕРАЦИЙ ДЛЯ НАХОЖДЕНИЯ
    ОПТИМАЛЬНОЙ СКОРОСТИ ОБУЧЕНИЯ" */
    inline int getMaxItersForLR() { return m_nMaxItersForLR; }
    void setMaxItersForLR(int value);
private:
    bool m_bConjugateGradient; /* используется ли метод сопряжённых градиентов
                                  для выбора оптимального направления на
                                  очередном шаге? */
    double m_maxLearningRate;// свойство "МАКС. ЗНАЧЕНИЕ СКОРОСТИ ОБУЧЕНИЯ"
    int m_nMaxItersForLR; /* свойство "МАКС.ЧИСЛО ИТЕРАЦИЙ ДЛЯ НАХОЖДЕНИЯ
                             ОПТИМАЛЬНОЙ СКОРОСТИ ОБУЧЕНИЯ" */

    double *m_aDirection;// вектор направления оптимизации весов сети
    double *m_aOldG;// старое значение суммарного вектора градиента
    CMultilayerPerceptron* m_pTempMLP;/*"временная" нейросеть, используемая
                                         как вспомогательная переменная при
                                         выборе длины шага по методу
                                         параболической интерполяции */

    /* Найти три начальные точки (lr1; etr1), (lr2; etr2) и (lr3; etr3), на
       основе которых будет осуществлятся поиск оптимального шага lr в заданном
       направлении m_aDirection[] (критерием оптимальности выступает ошибка
       обучения etr, которую надо минимизировать). Если такие точки найдены,
       возвращается true, в противном случае - false. */
    bool find_init_lrs(double& lr1, double& lr2, double& lr3,
                       double& etr1, double& etr2, double& etr3);

    /* Найти длину оптимального шага lr в заданном направлении m_aDirection[]
       по методу Брента (критерием оптимальности выступает ошибка обучения etr,
       которую надо минимизировать). В качестве стартовых точек метода Брента
       используются (lr1; etr1), (lr2; etr2) и (lr3; etr3). */
    void find_optimal_lr_by_brent(double lr1, double lr2, double lr3, double tol,
                                  double& lr, double& etr);

    /* Найти оптимальный шаг в заданном направлении m_aDirection по методу
       Брента. Найденное значение оптимального шага записывается в передаваемый
       по ссылке аргумент lr, а соответствующее значение целевой функции
       (функции ошибки) - в передаваемый по ссылке аргумент etr. */
    void find_optimal_learning_rate(double& lr, double& etr);

    /* Вычислить среднеквадратичную ошибку обучения как функцию от длины шага
       stepsize в направлении m_aDirection. */
    double calculate_training_error(double stepsize);
protected:
    // Выделить память для всех промежуточных переменных и инициализировать их
    void initialize_training();

    // Освободить память ото всех промежуточных переменных
    void finalize_training();

    // Обновить весовые коэффициенты обучаемой нейросети
    void change_weights(int nEpoch, TTrainingState& training_state);
};

#endif // ANNLIB_H
