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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>

#include <QFile>

#include "additional_unit.h"
#include "error_messages.h"
#include "trainer_unit.h"

static const char* g_szAboutTrainingStart   = "Обучение начато.";
static const char* g_szTrainSetSamples      = "Кол-во примеров обучающего "\
        "мн-ва:   %1";
static const char* g_szControlSetSamples    = "Кол-во примеров контрольного "\
        "мн-ва: %1";
static const char* g_szTrainedNetWeigths    = "Кол-во весов обучаемой "\
        "нейросети:   %1";

static const char* g_szEpochTitle               = "№ эпохи";
static const char* g_szTrainingErrorTitle       = "   Ошибка обучения";
static const char* g_szGeneralizationErrorTitle = "  Ошибка обобщения";

static const char* g_szStoppingByMaxEpochs  = "Обучение остановлено по "\
        "достижении максимального количества эпох.";
static const char* g_szEarlyStopping        = "Обучение остановлено "\
        "вследствие возрастания ошибки обобщения.";
static const char* g_szStoppingByGradient   = "Обучение остановлено по "\
        "причине слишком малого значения нормы градиента.";
static const char* g_szStoppingByMinError   = "Обучение остановлено из-за "\
        "невозможности дальнейшего уменьшения ошибки обучения.";
static const char* g_szStoppingByTrainingError = "Обучение остановлено по "\
        "достижении достаточно малого значения ошибки обучения.";
static const char* g_szStoppingByGeneralizationError = "Обучение остановлено "\
        "по достижении достаточно малого значения ошибки обобщения.";
static const char* g_szEpochDuration        = "Средняя продолжительность "\
        "одной эпохи обучения составила %1 сек.";
static const char* g_szEpochDurationDetails = "При этом собственно обучение "\
        "заняло %1 сек, а тестирование (вычисление ошибки) заняло %2 сек.";
static const char* g_szNewRestart = "Запущен %1-й рестарт процесса обучения.";

using namespace std;

/*****************************************************************************/
/* ОПЕРАЦИИ КЛАССА CTrainerForMLP */
/*****************************************************************************/

CTrainerForMLP::CTrainerForMLP():QObject(0)
{
    m_bCanTrain = false;

    m_sTrainedNetFilename = "";
    m_nRestarts = 1;

    m_bDoLogging = false;
    m_bShowTrainingError = false;
    m_bShowGeneralizationError = false;

    m_pTrainingAlgorithm = 0;

    m_aTrainInputs = 0; m_aTrainTargets = 0; m_nTrainSamples = 0;
    m_aControlInputs = 0; m_aControlTargets = 0; m_nControlSamples = 0;
    m_bControlSetIsLoaded = false;

    m_aGeneralizationErrors = 0;
    m_nMedfiltOrder = 1;
    m_aMedfiltBuffer = 0;
}

CTrainerForMLP::~CTrainerForMLP()
{
    if (m_pTrainingAlgorithm != 0)
    {
        delete m_pTrainingAlgorithm;
        m_pTrainingAlgorithm = 0;
    }

    if (m_aTrainInputs != 0)
    {
        delete[] m_aTrainInputs;
        m_aTrainInputs = 0;
    }
    if (m_aTrainTargets != 0)
    {
        delete[] m_aTrainTargets;
        m_aTrainTargets = 0;
    }

    if (m_bControlSetIsLoaded)
    {
        if (m_aControlInputs != 0)
        {
            delete[] m_aControlInputs;
            m_aControlInputs = 0;
        }
        if (m_aControlTargets != 0)
        {
            delete[] m_aControlTargets;
            m_aControlTargets = 0;
        }
        m_bControlSetIsLoaded = false;
    }

    if (m_aGeneralizationErrors != 0)
    {
        delete[] m_aGeneralizationErrors;
        m_aGeneralizationErrors = 0;
    }

    if (m_aMedfiltBuffer != 0)
    {
        delete[] m_aMedfiltBuffer;
        m_aMedfiltBuffer = 0;
    }
}

/* Инициализировать алгоритм обучения с помощью списка ключей запуска
rParams, сформированном путём анализа аргументов командной строки
(см. описание функции parse_command_line).
   В случае успешной инициализации возвращается true, а в случае ошибки,
вызванной указанием недопустимого ключа либо неверного значения допустимого
ключа, - false. */
bool CTrainerForMLP::initialize_params(const TCmdParams& rParams)
{
    bool result = true;
    m_goalError = 0.0;
    int nReadArgsCount = rParams.size();

    if (m_pTrainingAlgorithm != 0)
    {
        delete m_pTrainingAlgorithm;
        m_pTrainingAlgorithm = 0;
    }

    if (m_aTrainInputs != 0)
    {
        delete[] m_aTrainInputs;
        m_aTrainInputs = 0;
    }
    if (m_aTrainTargets != 0)
    {
        delete[] m_aTrainTargets;
        m_aTrainTargets = 0;
    }
    m_nTrainSamples = 0;

    if (m_bControlSetIsLoaded)
    {
        if (m_aControlInputs != 0)
        {
            delete[] m_aControlInputs;
            m_aControlInputs = 0;
        }
        if (m_aControlTargets != 0)
        {
            delete[] m_aControlTargets;
            m_aControlTargets = 0;
        }
        m_nControlSamples = 0;
        m_bControlSetIsLoaded = false;
    }
    else
    {
        m_aControlInputs = 0;
        m_aControlTargets = 0;
        m_nControlSamples = 0;
    }

    if (m_aGeneralizationErrors != 0)
    {
        delete[] m_aGeneralizationErrors;
        m_aGeneralizationErrors = 0;
    }
    if (m_aMedfiltBuffer != 0)
    {
        delete[] m_aMedfiltBuffer;
        m_aMedfiltBuffer = 0;
    }

    /* Загружаем нейронную сеть из файла. Имя файла определяется значением
    ключа "mlp". В случае ошибки выводим в stderr соответствующее сообщение
    и устанавливаем флаг возврата result в false. */
    if (rParams.contains("mlp"))
    {
        nReadArgsCount--;
        if (QFile::exists(rParams["mlp"]))
        {
            if (m_TrainedNet.load(rParams["mlp"]))
            {
                m_sTrainedNetFilename = rParams["mlp"];
            }
            else
            {
                result = false;
                cerr << qPrintable(QString(g_szMlpReadingError).arg(
                                       rParams["mlp"]));
            }
        }
        else
        {
            result = false;
            cerr << qPrintable(QString(g_szFileDoesNotExist).arg(
                                   rParams["mlp"]));
        }
    }
    else
    {
        cerr << qPrintable(QString(g_szArgIsNotFound).arg("mlp"));
        result = false;
    }

    if (result)
    {
        if (rParams.contains("goal"))
        {
            nReadArgsCount--;
            double goal = rParams["goal"].toFloat(&result);
            if (result)
            {
                if (goal >= 0.0)
                {
                    m_goalError = goal;
                }
                else
                {
                    cerr << qPrintable(QString(g_szIncorrectGoal));
                    result = false;
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szIncorrectGoal));
            }
        }
    }

    /* Загружаем обучающее множество из файла, название которого задано
    значением ключа "train". В случае ошибки - выводим сообщение в stderr
    и устанавливаем result = false. */
    if (result)
    {
        if (rParams.contains("train"))
        {
            nReadArgsCount--;

            int nTrainInputs = 0, nTrainTargets = 0, nTrainSamples = 0;
            if (QFile::exists(rParams["train"]))
            {
                if (load_trainset(rParams["train"], 0, 0, nTrainSamples, nTrainInputs, nTrainTargets))
                {
                    int iLastLayer = m_TrainedNet.getLayersCount() - 1;
                    if ((nTrainInputs != m_TrainedNet.getInputsCount())
                        || (nTrainTargets
                            != m_TrainedNet.getLayerSize(iLastLayer)))
                    {
                        result = false;
                        cerr<<qPrintable(QString(g_szTrainsetStructureError));
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szTrainsetReadingError).arg(
                                           rParams["train"]));
                    result = false;
                }
            }
            else
            {
                result = false;
                cerr << qPrintable(QString(g_szFileDoesNotExist).arg(
                                       rParams["train"]));
            }
            if (result)
            {
                m_nTrainSamples = nTrainSamples;
                m_aTrainInputs = new double[m_nTrainSamples * nTrainInputs];
                m_aTrainTargets = new double[m_nTrainSamples * nTrainTargets];
                if (!load_trainset(
                        rParams["train"], m_aTrainInputs, m_aTrainTargets,
                        nTrainSamples, nTrainInputs, nTrainTargets))
                {
                    cerr << qPrintable(QString(g_szTrainsetReadingError).arg(
                                           rParams["train"]));
                    result = false;
                }
            }
        }
        else
        {
            result = false;
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("train"));
        }
    }

    /* Если задан ключ "control", то загружаем контрольное множество из файла,
    название которого задано значением ключа "control". Если значение ключа
    "control" не является названием файла, то для формирования контрольного
    множества случайным образом выделяем часть обучающего множества. Размер
    этой части либо определяется автоматически (если ключ "control" вообще не
    имеет значения), либо задаётся пользователем (тогда значение ключа
    "control" должно задавать вещественное число больше нуля и меньше единицы
    - коэффициент разбиения).
       В случае ошибки - выводим сообщение в stderr и устанавливаем
    result = false. */
    if (result && rParams.contains("control"))
    {
        nReadArgsCount--;
        if (QFile::exists(rParams["control"]))
        {
            int nControlInputs = 0, nControlTargets = 0, nControlSamples = 0;
            if (load_trainset(rParams["control"], 0, 0, nControlSamples,
                              nControlInputs, nControlTargets))
            {
                int iLastLayer = m_TrainedNet.getLayersCount() - 1;
                if ((nControlInputs != m_TrainedNet.getInputsCount())
                    || (nControlTargets
                        != m_TrainedNet.getLayerSize(iLastLayer)))
                {
                    result = false;
                    cerr << qPrintable(QString(g_szControlsetStructureError));
                }
            }
            else
            {
                result = false;
                cerr << qPrintable(QString(g_szControlsetReadingError).arg(
                                       rParams["control"]));
            }
            if (result)
            {
                m_nControlSamples = nControlSamples;
                m_bControlSetIsLoaded = true;
                m_aControlInputs = new double[
                        m_nControlSamples * nControlInputs];
                m_aControlTargets = new double[
                        m_nControlSamples * nControlTargets];
                if (!load_trainset(
                        rParams["control"], m_aControlInputs,m_aControlTargets,
                        nControlSamples, nControlInputs, nControlTargets))
                {
                    result = false;
                    cerr << qPrintable(QString(g_szControlsetReadingError).arg(
                                           rParams["control"]));
                }
            }
        }
        else
        {
            if (!rParams["control"].isEmpty())
            {
                double r = rParams["control"].toFloat(&result);
                if (!result)
                {
                    cerr << qPrintable(QString(g_szFileDoesNotExist).arg(
                                           rParams["control"]));
                }
                else
                {
                    if ((r <= 0.0) || (r >= 1.0))
                    {
                        result = false;
                        cerr << qPrintable(
                                    QString(g_szSeparationFactorIncorrect));
                    }
                    else
                    {
                        if (!divide_between_train_and_control_sets(r))
                        {
                            cerr<< qPrintable(
                                       QString(g_szControlsetCannotBeCreated));
                            result = false;
                        }
                    }
                }
            }
            else // для контрольного множества "отщипываем" кусочек обучающего
            {
                if (!divide_between_train_and_control_sets())
                {
                    cerr << qPrintable(QString(g_szControlsetCannotBeCreated));
                    result = false;
                }
            }
        }
    }

    /* Если задан ключ "control", то проверяем наличие ключа "earlystop". Если
    данный ключ в наличии, то для определения момента завершения обучении будет
    использоваться критерий раннего останова. Если же данный ключ не обнаружен,
    то, соответственно, критерий раннего останова использоваться не будет.
       В случае ошибки (ключ "earlystop" задан, а ключ "control" не задан, т.е.
    нет контрольного множества, которое используется в методе раннего останова)
    - выводим сообщение в stderr и устанавливаем result = false. */
    if (result)
    {
        if (m_nControlSamples > 0)
        {
            if (rParams.contains("earlystop"))
            {
                nReadArgsCount--;
                if (!rParams["earlystop"].isEmpty())
                {
                    result = false;
                    cerr << qPrintable(QString(g_szImpossibleVal).arg(
                                           "earlystop"));
                }
                else
                {
                    m_bEarlyStopping = true;
                }
            }
            else
            {
                m_bEarlyStopping = false;
            }
        }
        else
        {
            if (rParams.contains("earlystop"))
            {
                result = false;
                cerr << qPrintable(QString(g_szSuperfluousArg).arg(
                                       "earlystop"));
            }
            else
            {
                m_bEarlyStopping = false;
            }
        }
    }

    /* Если задан ключ "earlystop", то проверяем наличие ключа "esmooth" и
    считываем его значение, которое определяет порядок медианного фильтра
    для сглаживания траектории ошибки обобщения. В случае ошибки (ключ не
    найден либо его значение не является целым положительным числом) - выводим
    сообщение в stderr и устанавливаем result = false. */
    if (result)
    {
        if (m_bEarlyStopping)
        {
            if (rParams.contains("esmooth"))
            {
                nReadArgsCount--;
                int nSmooth = rParams["esmooth"].toInt(&result);
                if (result)
                {
                    if (nSmooth > 0)
                    {
                        m_nMedfiltOrder = nSmooth;
                    }
                    else
                    {
                        result = false;
                        cerr << qPrintable(QString(g_szIncorrectMedfiltOrder));
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szIncorrectMedfiltOrder));
                }
            }
            else
            {
                result = false;
                cerr << qPrintable(QString(g_szArgIsNotFound).arg("esmooth"));
            }
        }
        else
        {
            if (rParams.contains("esmooth"))
            {
                result = false;
                cerr << qPrintable(QString(g_szSuperfluousArg).arg("esmooth"));
            }
        }
    }

    /* Читаем значение ключа "alg", определяющее алгоритм обучения. Возможные
    значения ключа: "bp_s" (стохастическое обратное распространение), "dbd"
    (Incremental Delta Bar Delta), "bp_b" (пакетное обратное распространение -
    метод наискорейшего спуска), "rp" ("упругое" обратное распространение),
    "cg" (алгоритм сопряжённых градиентов).
       Если ключ "alg" не найден или значение этого ключа не относится ко
    вышеуказанному множеству допустимых значений, то в stderr выводится
    сообщение о соответствующей ошибке, а флаг возврата result устанавливается
    в false.*/
    if (result)
    {
        if (rParams.contains("alg"))
        {
            nReadArgsCount--;
            if (!rParams["alg"].isEmpty())
            {
                if (rParams["alg"].compare("bp_s", Qt::CaseInsensitive) == 0)
                {
                    m_pTrainingAlgorithm = new COnlineBackpropTraining();
                    qobject_cast<COnlineBackpropTraining*>
                            (m_pTrainingAlgorithm)->setAdaptiveRate(false);
                }
                else if (rParams["alg"].compare("dbd",Qt::CaseInsensitive)== 0)
                {
                    m_pTrainingAlgorithm = new COnlineBackpropTraining();
                    qobject_cast<COnlineBackpropTraining*>
                            (m_pTrainingAlgorithm)->setAdaptiveRate(true);
                }
                else if (rParams["alg"].compare("bp_b",Qt::CaseInsensitive)==0)
                {
                    m_pTrainingAlgorithm = new CGradientDescentTraining();
                    qobject_cast<CGradientDescentTraining*>
                           (m_pTrainingAlgorithm)->setConjugateGradient(false);
                }
                else if (rParams["alg"].compare("rp",Qt::CaseInsensitive) == 0)
                {
                    m_pTrainingAlgorithm = new CResilientBackpropTraining();
                }
                else if (rParams["alg"].compare("cg",Qt::CaseInsensitive) == 0)
                {
                    m_pTrainingAlgorithm = new CGradientDescentTraining();
                    qobject_cast<CGradientDescentTraining*>
                           (m_pTrainingAlgorithm)->setConjugateGradient(true);
                }
                else
                {
                    result = false;
                    cerr << qPrintable(QString(g_szTrainingAlgIncorrect));
                }
                if (result)
                {
                    QObject::connect(
                            this, SIGNAL(infoAboutTrainingStopping()),
                            m_pTrainingAlgorithm, SLOT(stop_training_state()));
                    QObject::connect(
                            m_pTrainingAlgorithm, SIGNAL(start_training()),
                            this, SLOT(start_training()));
                    QObject::connect(
                            m_pTrainingAlgorithm, SIGNAL(do_training_epoch(int)),
                            this, SLOT(do_training_epoch(int)));
                    QObject::connect(
                            m_pTrainingAlgorithm,
                            SIGNAL(end_training(TTrainingState)),
                            this, SLOT(end_training(TTrainingState)));
                }
            }
            else
            {
                result = false;
                cerr << qPrintable(QString(g_szTrainingAlgIncorrect));
            }
        }
        else
        {
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("alg"));
            result = false;
        }
    }

    /* Проверяем, надо ли создавать протокол обучения. Если ключ "log" есть,
    то надо, если же ключ "log" отсутствует, то не надо. */
    if (result)
    {
        if (rParams.contains("log"))
        {
            nReadArgsCount--;
            if (!rParams["log"].isEmpty())
            {
                result = false;
                cerr << qPrintable(QString(g_szImpossibleVal).arg("log"));
            }
            else
            {
                m_bDoLogging = true;
            }
        }
        else
        {
            m_bDoLogging = false;
        }
    }

    /* Если указано, что процесс обучения надо протоколировать, то проверяем,
    надо ли протоколировать траекторию ошибки обучения. Если ключ "etr" в
    наличии, то надо, если же данный ключ не указан, то не надо. */
    if (result)
    {
        if (m_bDoLogging)
        {
            if (rParams.contains("etr"))
            {
                nReadArgsCount--;
                if (!rParams["etr"].isEmpty())
                {
                    result = false;
                    cerr << qPrintable(QString(g_szImpossibleVal).arg("etr"));
                }
                else
                {
                    m_bShowTrainingError = true;
                }
            }
            else
            {
                m_bShowTrainingError = false;
            }
        }
        else
        {
            if (rParams.contains("etr"))
            {
                result = false;
                cerr << qPrintable(QString(g_szSuperfluousArg).arg("etr"));
            }
        }
    }

    /* Если указано, что процесс обучения надо протоколировать, то проверяем,
    надо ли протоколировать траекторию ошибки обобщения. Если ключ "eg" в
    наличии, то надо, если же данный ключ не указан, то не надо. */
    if (result)
    {
        if (m_bDoLogging)
        {
            if (rParams.contains("eg"))
            {
                if (m_nControlSamples > 0)
                {
                    nReadArgsCount--;
                    if (!rParams["eg"].isEmpty())
                    {
                        result = false;
                        cerr << qPrintable(QString(g_szImpossibleVal).arg("eg"));
                    }
                    else
                    {
                        m_bShowGeneralizationError = true;
                    }
                }
                else
                {
                    result = false;
                    cerr << qPrintable(QString(g_szSuperfluousArg).arg("eg"));
                }
            }
            else
            {
                m_bShowGeneralizationError = false;
            }
        }
        else
        {
            if (rParams.contains("eg"))
            {
                result = false;
                cerr << qPrintable(QString(g_szSuperfluousArg).arg("eg"));
            }
        }
    }

    /* Читаем значение ключа "maxepochs", определяющее максимально допустимое
    число эпох обучения. Если ключ "maxepochs" не найден или значение этого
    ключа не является положительным целым числом, то в stderr выводится
    сообщение о соответствующей ошибке, а флаг возврата result устанавливается
    в false. */
    if (result)
    {
        if (rParams.contains("maxepochs"))
        {
            nReadArgsCount--;
            int nMaxEpochs = rParams["maxepochs"].toInt(&result);
            if (result)
            {
                if (nMaxEpochs > 0)
                {
                    m_pTrainingAlgorithm->setMaxEpochsCount(nMaxEpochs);
                    if (m_bEarlyStopping || m_bShowGeneralizationError)
                    {
                        m_aGeneralizationErrors = new double[nMaxEpochs + 1];
                    }
                }
                else
                {
                    result = false;
                    cerr << qPrintable(QString(g_szMaxEpochsIncorrect));
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szMaxEpochsIncorrect));
            }
        }
        else
        {
            result = false;
            cerr << qPrintable(QString(g_szArgIsNotFound).arg("maxepochs"));
        }
    }

    /* Если в алгоритме обучения используется ранний останов, то необходимо
    сделать ещё одну проверку следующего характера.
       Порядок медианного фильтра для сглаживания траектории ошибки обобщения
    должен быть меньше, чем максимально допустимое число эпох обучения.
    Проверяем это условие. Если оно не выполняется, то в stderr выводим
    сообщение о соответствующей ошибке, а флаг возврата result устанавливаем
    в false. */
    if (result && m_bEarlyStopping)
    {
        if (m_pTrainingAlgorithm->getMaxEpochsCount() <= m_nMedfiltOrder)
        {
            result = false;
            cerr << qPrintable(QString(g_szMedfiltOrderIsVeryLarge));
        }
    }

    /* Читаем значение ключа "restarts", определяющее максимально допустимое
    число эпох обучения. Если ключ "restarts" не найден, то используется
    значение по умлочанию, равное единице. Если же ключ "restarts" найден, но
    его значение не является положительным целым числом, то в stderr выводится
    сообщение о соответствующей ошибке, а флаг возврата result устанавливается
    в false. */
    if (result)
    {
        if (rParams.contains("restarts"))
        {
            nReadArgsCount--;
            int nRestarts = rParams["restarts"].toInt(&result);
            if (result)
            {
                if (nRestarts > 0)
                {
                    m_nRestarts = nRestarts;
                }
                else
                {
                    result = false;
                    cerr << qPrintable(QString(g_szRestartsIncorrect));
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szRestartsIncorrect));
            }
        }
        else
        {
            m_nRestarts = 1;
        }
    }

    // Если алгоритм обучения основан на пакетном вычислении градиента, то...
    if (qobject_cast<CBatchBackpropTraining*>(m_pTrainingAlgorithm) != 0)
    {
        /* Читаем значение ключа "eps", которое является точностью метода
        сопряжённых градиентов (если норма суммарного градиента после очередной
        эпохи стала меньше, чем исходная норма суммарного градиента, умноженная
        на данное число, то обучение завершается). */
        if (result)
        {
            if (rParams.contains("eps"))
            {
                nReadArgsCount--;
                double eps = rParams["eps"].toFloat(&result);
                if (result)
                {
                    if ((eps > 0.0) && (eps <= 1.0))
                    {
                        qobject_cast<CBatchBackpropTraining*>
                                (m_pTrainingAlgorithm)->setEpsilon(eps);
                    }
                    else
                    {
                        cerr << qPrintable(QString(g_szGradientEpsIncorrect));
                        result = false;
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szGradientEpsIncorrect));
                }
            }
            else
            {
                cerr << qPrintable(QString(g_szArgIsNotFound).arg("eps"));
                result = false;
            }
        }

        /* Если алгоритм обучения - это алгоритм сопряжённых градиентов или
           пакетного наискорейшего спуска, то... */
        if (qobject_cast<CGradientDescentTraining*>(m_pTrainingAlgorithm) != 0)
        {
            CGradientDescentTraining* pAlg
                    = qobject_cast<CGradientDescentTraining*>
                      (m_pTrainingAlgorithm);
            /* Читаем значение ключа "lr_max", которое является максимальным
            значением коэффициента скорости обучения (используется как
            ограничение сверху в процедуре поиска оптимального значения
            коэффициента скорости обучения).
               Если ключ "lr_max" не найден или значение этого ключа не
            является положительным вещественным числом, то в stderr выводится
            сообщение о соответствующей ошибке, а флаг возврата result
            устанавливается в false. */
            if (result)
            {
                if (rParams.contains("lr_max"))
                {
                    nReadArgsCount--;
                    double learning_rate = rParams["lr_max"].toFloat(&result);
                    if (result)
                    {
                        if (learning_rate > 0.0)
                        {
                            pAlg->setMaxLearningRate(learning_rate);
                        }
                        else
                        {
                            cerr << qPrintable(
                                        QString(g_szMaxLearningRateIncorrect));
                            result = false;
                        }
                    }
                    else
                    {
                        cerr << qPrintable(
                                    QString(g_szMaxLearningRateIncorrect));
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szArgIsNotFound).arg(
                                           "lr_max"));
                    result = false;
                }
            }

            /* Читаем значение ключа "lr_iters", которое является максимальным
            числом итераций алгоритма нахождения оптимальной скорости обучения
            на каждой эпохе.
               Если ключ "lr_iters" не найден или значение этого ключа не
            является положительным целым числом, то в stderr выводится
            сообщение о соответствующей ошибке, а флаг возврата result
            устанавливается в false. */
            if (result)
            {
                if (rParams.contains("lr_iters"))
                {
                    nReadArgsCount--;
                    int iterations = rParams["lr_iters"].toInt(&result);
                    if (result)
                    {
                        if (iterations > 0)
                        {
                            pAlg->setMaxItersForLR(iterations);
                        }
                        else
                        {
                            QString sErr(g_szMaxLearningRateItersIncorrect);
                            cerr << qPrintable(sErr);
                            result = false;
                        }
                    }
                    else
                    {
                        QString sErr(g_szMaxLearningRateItersIncorrect);
                        cerr << qPrintable(sErr);
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szArgIsNotFound).arg(
                                           "lr_iters"));
                    result = false;
                }
            }
        }
        else
        {
            CResilientBackpropTraining* pAlg
                    = qobject_cast<CResilientBackpropTraining*>
                      (m_pTrainingAlgorithm);
            /* Читаем значение ключа "lr", которое является начальным
            значением коэффициента скорости обучения (не меньше 10^-6 и не
            больше 50).
               Если ключ "lr" не найден или значение этого ключа не входит
            в вышеуказаннЫй диапазон, то в stderr выводится сообщение о
            соответствующей ошибке, а флаг возврата result устанавливается в
            false. */
            if (result)
            {
                if (rParams.contains("lr"))
                {
                    nReadArgsCount--;
                    double learning_rate = rParams["lr"].toFloat(&result);
                    if (result)
                    {
                        if ((learning_rate >= pAlg->getMinLearningRate())
                            && (learning_rate <= pAlg->getMaxLearningRate()))
                        {
                            pAlg->setInitialLearningRate(learning_rate);
                        }
                        else
                        {
                            cerr << qPrintable(
                                        QString(g_szLearningRateIncorrect));
                            result = false;
                        }
                    }
                    else
                    {
                        cerr << qPrintable(QString(g_szLearningRateIncorrect));
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szArgIsNotFound).arg("lr"));
                    result = false;
                }
            }
        }
    }
    /* Если алгоритм обучения основан на стохастическом вычислении градиента,
    то... */
    else
    {
        if (result)
        {
            COnlineBackpropTraining* pAlg
                    = qobject_cast<COnlineBackpropTraining*>
                      (m_pTrainingAlgorithm);

            if (pAlg->getAdaptiveRate()) // стохастический Delta-Bar-Delta
            {
                /* Читаем значение ключа "lr", которое является начальным
                значением коэффициента скорости обучения (не меньше 10^-6 и не
                больше 50).
                   Если ключ "lr" не найден или значение этого ключа не
                является положительным числом, то в stderr выводится сообщение
                о соответствующей ошибке, а флаг возврата result
                устанавливается в false. */
                if (rParams.contains("lr"))
                {
                    nReadArgsCount--;
                    double learning_rate = rParams["lr"].toFloat(&result);
                    if (result)
                    {
                        if (learning_rate > 0.0)
                        {
                            pAlg->setStartLearningRateParam(learning_rate);
                            pAlg->setFinalLearningRateParam(learning_rate);
                        }
                        else
                        {
                            cerr << qPrintable(
                                        QString(g_szLearningRateIncorrect));
                            result = false;
                        }
                    }
                    else
                    {
                        cerr << qPrintable(QString(g_szLearningRateIncorrect));
                    }
                }
                else
                {
                    cerr << qPrintable(QString(g_szArgIsNotFound).arg("lr"));
                    result = false;
                }

                /* Читаем значение ключа "theta", которое является значением
                управляющего параметра "theta".
                   Если ключ "theta" не найден или значение этого ключа н
                является положительным числом, то в stderr выводится сообщение
                о соответствующей ошибке, а флаг возврата result
                устанавливается в false. */
                if (result)
                {
                    if (rParams.contains("theta"))
                    {
                        nReadArgsCount--;
                        double theta = rParams["theta"].toFloat(&result);
                        if (result)
                        {
                            if (theta > 0.0)
                            {
                                pAlg->setTheta(theta);
                            }
                            else
                            {
                                cerr<< qPrintable(QString(g_szIncorrectTheta));
                                result = false;
                            }
                        }
                        else
                        {
                            cerr << qPrintable(QString(g_szIncorrectTheta));
                        }
                    }
                    else
                    {
                        cerr << qPrintable(QString(g_szArgIsNotFound).arg(
                                               "theta"));
                        result = false;
                    }
                }
            }
            else // стохастическое обратное распространение ошибки
            {
                /* Читаем значение ключа "lr", которое является значением
                коэффициента скорости обучения (положительное вещественное
                число).
                   Если значение ключа "lr" не является положительным
                вещественным числом, то в stderr выводится сообщение о
                соответствующей ошибке, а флаг возврата result
                устанавливается в false.
                   Если ключ "lr" не найден, то читаем значения ключей
                "lr_init" и "lr_fin", являющихся значениями коэффициента
                скорости обучения на первой и последней эпохах обучения
                соответственно. Текущий коэффициент скорости обучения
                изменяется на каждой эпохе от "lr_init" до "lr_fin" по
                квадратичному закону.
                   Если ключи "lr_init" и "lr_fin" не найдены или их значения
                не являются положительными вещественными числами, то в stderr
                выводится сообщение о соответствующей ошибке, а флаг возврата
                result устанавливается в false. */
                if (result)
                {
                    if (rParams.contains("lr"))
                    {
                        nReadArgsCount--;
                        double lr = rParams["lr"].toFloat(&result);
                        if (result)
                        {
                            if (lr > 0.0)
                            {
                                pAlg->setStartLearningRateParam(lr);
                                pAlg->setFinalLearningRateParam(lr);
                            }
                            else
                            {
                                cerr<< qPrintable(
                                           QString(g_szLearningRateIncorrect));
                                result = false;
                            }
                        }
                        else
                        {
                            cerr<< qPrintable(
                                       QString(g_szLearningRateIncorrect));
                            result = false;
                        }
                    }
                    else
                    {
                        if (rParams.contains("lr_init")
                            && rParams.contains("lr_fin"))
                        {
                            nReadArgsCount -= 2;
                            bool ok1 = true, ok2 = true;
                            double lr_init = rParams["lr_init"].toFloat(&ok1);
                            double lr_fin = rParams["lr_fin"].toFloat(&ok2);
                            if (ok1 && ok2)
                            {
                                if ((lr_init > 0.0) && (lr_fin > 0.0))
                                {
                                    pAlg->setStartLearningRateParam(lr_init);
                                    pAlg->setFinalLearningRateParam(lr_fin);
                                }
                                else
                                {
                                    QString sErr(g_szLearningRateIncorrect);
                                    cerr << qPrintable(sErr);
                                    result = false;
                                }
                            }
                            else
                            {
                                QString sErr(g_szLearningRateIncorrect);
                                cerr << qPrintable(sErr);
                                result = false;
                            }
                        }
                        else
                        {
                            QString sErr(g_szArgIsNotFound);
                            if (rParams.contains("lr_init")
                                && (!rParams.contains("lr_fin")))
                            {
                                cerr << qPrintable(sErr.arg("lr_fin"));
                            }
                            else if ((!rParams.contains("lr_init"))
                                && rParams.contains("lr_fin"))
                            {
                                cerr << qPrintable(sErr.arg("lr_init"));
                            }
                            else
                            {
                                cerr << qPrintable(sErr.arg("lr"));
                            }
                            result = false;
                        }
                    }
                }
            }
        }
    }

    /* Мы проверили все ключи, которые пользователь может указать при запуске
    программы в режиме обучения. Если остались ещё непроверенные ключи, то
    в stderr выводим соответствующее сообщение, а флаг возврата result
    устанавливаем в false, поскольку данная ситация рассматривается как
    ошибка. */
    if (result)
    {
        if (nReadArgsCount > 0)
        {
            result = false;
            cerr << qPrintable(QString(g_szManyArgs));
        }
    }

    m_bCanTrain = result;

    return result;
}

/* Обучить нейросеть. Сама сеть и обучающее множество должны быть
предварительно заданы (см. описание операции set_param). Остальные
параметры алгоритма обучения также могут быть заданы перед вызовом
операции train с помощью вышеупомянутой операции set_param (если
они не были заданы, то используются значения по умолчанию).
   Если не задана обучаемая сеть либо обучающее множество, то функция
возвращает false и выводит соответствующее сообщение об ошибке. Если
всё нормально, то функция возвращает true. */
bool CTrainerForMLP::do_training()
{
    bool result;
    if (m_bCanTrain)
    {
        m_minError = FLT_MAX;
        for (int i = 1; i <= m_nRestarts; i++)
        {
            cout << endl << endl;
            cout << qPrintable(QString(g_szNewRestart).arg(i));
            cout << endl << endl;

            m_nEpochsCount = 0;

            m_ControlDuration = 0;
            m_TrainingDuration = 0;

            m_bStoppedByCriterion = false;
            m_bStoppedByGoal = false;

            m_pTrainingAlgorithm->train(
                    &m_TrainedNet, m_aTrainInputs, m_aTrainTargets,
                    m_nTrainSamples);
            m_TrainedNet.initialize_weights();
        }
    }
    else
    {
        result = false;
    }
    return result;
}

/* Вычислить ошибку обобщения после m_nEpochsCount-й эпохи обучения */
void CTrainerForMLP::calc_generalization_error()
{
    time_t start_time = time(0);

    m_aGeneralizationErrors[m_nEpochsCount] = m_TrainedNet.calculate_mse(
            m_aControlInputs, m_aControlTargets, m_nControlSamples);

    double duration = difftime(time(0), start_time);
    m_ControlDuration += duration;
}

bool CTrainerForMLP::check_early_stopping()
{
    if (m_nMedfiltOrder <= 0)
    {
        return false;
    }
    if ((m_nEpochsCount + 1) < m_nMedfiltOrder)
    {
        return false;
    }

    bool result;
    int i, j, k, iErr = m_nEpochsCount + 1 - m_nMedfiltOrder;
    m_aMedfiltBuffer[0] = m_aGeneralizationErrors[iErr];
    for (i = 1; i < m_nMedfiltOrder; i++)
    {
        if (m_aMedfiltBuffer[i-1] <= m_aGeneralizationErrors[iErr + i])
        {
            m_aMedfiltBuffer[i] = m_aGeneralizationErrors[iErr + i];
        }
        else
        {
            j = i - 2;
            while (j >= 0)
            {
                if (m_aMedfiltBuffer[j] <= m_aGeneralizationErrors[iErr + i])
                    break;
                else
                    j--;
            }
            j++;
            for (k = i; k > j; k--)
            {
                m_aMedfiltBuffer[k] = m_aMedfiltBuffer[k - 1];
            }
            m_aMedfiltBuffer[j] = m_aGeneralizationErrors[iErr + i];
        }
    }
    if ((m_nEpochsCount + 1) > m_nMedfiltOrder)
    {
        result = (m_aMedfiltBuffer[m_nMedfiltOrder / 2] > m_oldMedian);
    }
    else
    {
        result = false;
    }
    m_oldMedian = m_aMedfiltBuffer[m_nMedfiltOrder / 2];
    return result;
}

/* Поделить уже сформированное обучающее множество на два: собственно
обучающее и контрольное (для проверки критерия раннего останова).
   Определение доли примеров, отведённых для контрольного множества,
осуществляется на основе информации о числе свободных параметров обучаемой
нейронной сети, поэтому нейросеть также должна быть уже загружена.
   Тем не менее, доля примеров контрольного множества должна составлять
не менее 1% от общего числа примеров. */
bool CTrainerForMLP::divide_between_train_and_control_sets()
{
    double partOfControlset
            = (sqrt(2.0 * m_TrainedNet.getAllWeightsCount() - 1.0) - 1.0)
              / (2.0 * (m_TrainedNet.getAllWeightsCount()));
    if (partOfControlset < 0.01)
    {
        partOfControlset = 0.01;
    }
    return divide_between_train_and_control_sets(partOfControlset);
}

/* Поделить уже сформированное обучающее множество на два: собственно
обучающее и контрольное (для проверки критерия раннего останова).
   Доля примеров, отведённых для контрольного множества, задаётся
коэффициентом r. */
bool CTrainerForMLP::divide_between_train_and_control_sets(const double& r)
{
    //cerr << "divide_between_train_and_control_sets - begin\n";// for debug
    int nControlSamples = (int)floor(r * m_nTrainSamples);
    int nNewTrainSamples = m_nTrainSamples - nControlSamples;
    if ((nControlSamples <= 0) || (nNewTrainSamples <= 0))
    {
        cerr << g_szControlsetCannotBeCreated;
        return false;
    }

    int i, j;
    bool result = true;
    int *aTrainIndexes = 0;
    double *aTempTrainSample = 0;
    int nTrainInputs = m_TrainedNet.getInputsCount();
    int nTrainTargets = m_TrainedNet.getLayerSize(
            m_TrainedNet.getLayersCount() - 1);
    size_t nTrainInputSize = nTrainInputs * sizeof(double);
    size_t nTrainTargetSize = nTrainTargets * sizeof(double);
    try
    {
        aTempTrainSample = new double[nTrainInputs + nTrainTargets];
        //cerr << "aTempTrainSample = new double[nTrainInputs + nTrainTargets];\n"; // for debug
        aTrainIndexes = new int[m_nTrainSamples];
        //cerr << "aTrainIndexes = new int[m_nTrainSamples];\n"; // for debug
        calculate_rand_indexes(aTrainIndexes, m_nTrainSamples);
        //cerr << "calculate_rand_indexes(aTrainIndexes, m_nTrainSamples);\n"; // for debug
        for (i = 0; i < m_nTrainSamples; i++)
        {
            j = aTrainIndexes[i];

            memcpy(&aTempTrainSample[0], &m_aTrainInputs[i * nTrainInputs],
                   nTrainInputSize);
            memcpy(&m_aTrainInputs[i * nTrainInputs],
                   &m_aTrainInputs[j * nTrainInputs], nTrainInputSize);
            memcpy(&m_aTrainInputs[j * nTrainInputs], &aTempTrainSample[0],
                   nTrainInputSize);

            memcpy(&aTempTrainSample[0], &m_aTrainTargets[i * nTrainTargets],
                   nTrainTargetSize);
            memcpy(&m_aTrainTargets[i * nTrainTargets],
                   &m_aTrainTargets[j * nTrainTargets], nTrainTargetSize);
            memcpy(&m_aTrainTargets[j * nTrainTargets], &aTempTrainSample[0],
                   nTrainTargetSize);
        }
        m_nTrainSamples = nNewTrainSamples;
        m_nControlSamples = nControlSamples;
        m_aControlInputs = &m_aTrainInputs[m_nTrainSamples * nTrainInputs];
        m_aControlTargets = &m_aTrainTargets[m_nTrainSamples * nTrainTargets];
        if (aTempTrainSample != 0)
        {
            delete[] aTempTrainSample;
            aTempTrainSample = 0;
        }
        if (aTrainIndexes != 0)
        {
            delete[] aTrainIndexes;
            aTrainIndexes = 0;
        }
    }
    catch(...)
    {
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
    //cerr << "divide_between_train_and_control_sets - end\n";//for debug
    return result;
}

/* Пересчитать длительность обучения, добавив к текущему значению
m_TrainingStartTime время, затраченное на в_полнение последней эпохи
обучения. */
void CTrainerForMLP::update_training_duration()
{
    double epoch_duration = difftime(time(0), m_TrainingStartTime);
    m_TrainingDuration += epoch_duration;
}

/* Выполнить инициализацию некоторых атрибутов перед началом обучения, а
также занести в протокол обучения (если он ведётся) информацию о старте
процесса обучения. */
void CTrainerForMLP::start_training()
{
    m_nEpochsCount = 0;

    m_ControlDuration = 0.0;
    m_TrainingDuration = 0.0;

    m_bStoppedByCriterion = false;
    m_bStoppedByGoal = false;

    if (m_aMedfiltBuffer != 0)
    {
        delete[] m_aMedfiltBuffer;
        m_aMedfiltBuffer = 0;
    }
    if (m_bEarlyStopping && (m_nMedfiltOrder >= 1))
    {        
        m_aMedfiltBuffer = new double[m_nMedfiltOrder];
    }

    double cur_error = m_minError;

    if (m_bDoLogging)
    {
        QString sInfoMsg;
        double training_error = 0.0;

        sInfoMsg = QString(g_szAboutTrainingStart);
        cout << qPrintable(sInfoMsg) << endl;

        sInfoMsg = QString(g_szTrainedNetWeigths);
        cout << qPrintable(sInfoMsg.arg(m_TrainedNet.getAllWeightsCount()))
             << endl;

        sInfoMsg = QString(g_szTrainSetSamples);
        cout << qPrintable(sInfoMsg.arg(m_nTrainSamples)) << endl;

        if (m_nControlSamples > 0)
        {
            sInfoMsg = QString(g_szControlSetSamples);
            cout << qPrintable(sInfoMsg.arg(m_nControlSamples)) << endl;
        }

        if (m_bShowTrainingError)
        {
            training_error = m_TrainedNet.calculate_mse(
                    m_aTrainInputs, m_aTrainTargets, m_nTrainSamples);
        }

        if (m_bEarlyStopping || m_bShowGeneralizationError)
        {
            calc_generalization_error();
            cur_error = m_aGeneralizationErrors[0];
        }
        else
        {
            if (m_bShowTrainingError)
            {
                cur_error = training_error;
            }
        }

        sInfoMsg = QString(g_szEpochTitle);
        cout << endl << qPrintable(sInfoMsg);
        if (m_bShowTrainingError)
        {
            sInfoMsg = QString(g_szTrainingErrorTitle);
            cout << qPrintable(sInfoMsg);
        }
        if (m_bShowGeneralizationError)
        {
            sInfoMsg = QString(g_szGeneralizationErrorTitle);
            cout << qPrintable(sInfoMsg);
        }
        cout << endl;

        cout.setf(ios_base::scientific, ios_base::floatfield);

        cout.fill(' ');
        cout.width(strlen(g_szEpochTitle));
        cout << m_nEpochsCount;

        if (m_bShowTrainingError)
        {
            cout.fill(' ');
            cout.precision(5);
            cout.width(strlen(g_szTrainingErrorTitle));
            cout << training_error;
        }

        if (m_bShowGeneralizationError)
        {
            cout.fill(' ');
            cout.precision(5);
            cout.width(strlen(g_szGeneralizationErrorTitle));
            cout << m_aGeneralizationErrors[0];
        }

        cout << endl;
    }
    else
    {
        if (m_bEarlyStopping)
        {
            calc_generalization_error();
            cur_error = m_aGeneralizationErrors[0];
        }
    }

    if (cur_error < m_minError)
    {
        m_minError = cur_error;
        if (!m_TrainedNet.save(m_sTrainedNetFilename))
        {
            cerr << qPrintable(QString(g_szMlpWritingError).arg(
                                   m_sTrainedNetFilename));
        }
    }

    m_TrainingStartTime = time(0);
}

/* Занести в протокол обучения (если он ведётся) информацию об окончании
очередной эпохи обучения, а также проверить критерий раннего останова.
Если в соответствии с этим критерием обучение необходимо останавливать,
то алгоритму обучения посылается сигнал infoAboutTrainingStopping(). */
void CTrainerForMLP::do_training_epoch(int nEpochsCount)
{
    bool can_continue = true;
    double training_error = 0.0;

    m_nEpochsCount = nEpochsCount;
    update_training_duration();

    if (m_bEarlyStopping || m_bShowGeneralizationError)
    {
        calc_generalization_error();
    }

    if (m_bEarlyStopping)
    {
        if (check_early_stopping())
        {
            can_continue = false;
            m_bStoppedByCriterion = true;
        }
    }

    if (m_bDoLogging)
    {
        cout.fill(' ');
        cout.width(strlen(g_szEpochTitle));
        cout << m_nEpochsCount;

        if (m_bShowTrainingError)
        {
            training_error = m_TrainedNet.calculate_mse(
                    m_aTrainInputs, m_aTrainTargets, m_nTrainSamples);
            cout.fill(' ');
            cout.precision(5);
            cout.width(strlen(g_szTrainingErrorTitle));
            cout << training_error;
        }

        if (m_bShowGeneralizationError)
        {
            cout.fill(' ');
            cout.precision(5);
            cout.width(strlen(g_szGeneralizationErrorTitle));
            cout << m_aGeneralizationErrors[m_nEpochsCount];
        }

        cout << endl;
    }

    if (can_continue && (m_bEarlyStopping || m_bShowGeneralizationError
                         || m_bShowTrainingError))
    {
        if (m_bEarlyStopping || m_bShowGeneralizationError)
        {
            if (m_aGeneralizationErrors[m_nEpochsCount] <= m_goalError)
            {
                can_continue = false;
                m_bStoppedByGoal = true;
            }
        }
        else
        {
            if (training_error <= m_goalError)
            {
                can_continue = false;
                m_bStoppedByGoal = true;
            }
        }
    }

    bool can_save;
    if (m_bEarlyStopping || m_bShowGeneralizationError)
    {
        if (m_aGeneralizationErrors[m_nEpochsCount] < m_minError)
        {
            m_minError = m_aGeneralizationErrors[m_nEpochsCount];
            can_save = true;
        }
        else
        {
            can_save = false;
        }
    }
    else
    {
        if (m_bShowTrainingError)
        {
            if (training_error < m_minError)
            {
                m_minError = training_error;
                can_save = true;
            }
            else
            {
                can_save = false;
            }
        }
        else
        {
            can_save = true;
        }
    }

    if (can_save)
    {
        if (!m_TrainedNet.save(m_sTrainedNetFilename))
        {
            cerr << qPrintable(QString(g_szMlpWritingError).arg(
                                   m_sTrainedNetFilename));
        }
    }

    if (can_continue)
    {
        m_TrainingStartTime = time(0);
    }
    else
    {
        emit infoAboutTrainingStopping();
    }
}

/* Занести в протокол обучения (если он ведётся) информацию о завершении
процесса обучения. */
void CTrainerForMLP::end_training(TTrainingState state)
{
    if (m_bDoLogging)
    {
        QString sInfoMsg;
        cout << endl;
        switch (state)
        {
        case tsGRADIENT:
            sInfoMsg = QString(g_szStoppingByGradient);
            break;
        case tsMAXEPOCHS:
            sInfoMsg = QString(g_szStoppingByMaxEpochs);
            break;
        case tsMINIMUM:
            sInfoMsg = QString(g_szStoppingByMinError);
            break;
        default:
            if (m_bStoppedByCriterion)
            {
                sInfoMsg = QString(g_szEarlyStopping);
            }
            else if (m_bStoppedByGoal)
            {
                if (m_bShowGeneralizationError)
                {
                    sInfoMsg = QString(g_szStoppingByGeneralizationError);
                }
                else
                {
                    sInfoMsg = QString(g_szStoppingByTrainingError);
                }
            }
        }
        cout << qPrintable(sInfoMsg) << endl;

        double mean_training_duration = m_TrainingDuration / m_nEpochsCount;

        double mean_control_duration = m_ControlDuration / (m_nEpochsCount+1);

        sInfoMsg = QString(g_szEpochDuration).arg(
                    mean_training_duration + mean_control_duration);
        cout << qPrintable(sInfoMsg) << endl;
        sInfoMsg = QString(g_szEpochDurationDetails).arg(
                    mean_training_duration).arg(mean_control_duration);
        cout << qPrintable(sInfoMsg) << endl;
    }
}

