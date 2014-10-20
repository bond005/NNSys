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

#include "error_messages.h"

const char *g_szUnknownError = "Неизвестная ошибка.\n";
const char *g_szNoArgs = "При запуске программы не были указаны аргументы "\
        "командной строки.\n";
const char *g_szFewArgs = "Указано слишком мало аргументов командной строки.\n";
const char *g_szManyArgs = "Указано слишком много аргументов командной строки.\n";
const char *g_szImpossibleVal = "Параметр с ключом \"%1\" не должен иметь "\
        "никакого значения.\n";
const char *g_szNullVal = "Параметр с ключом \"%1\" должен иметь некоторое "\
        "значение.\n";
const char *g_szSuperfluousArg = "Параметр с ключом \"%1\" является лишним в "\
        "данном контексте.\n";
const char *g_szIncorrectExecutionMode = "Не удалось определить режим "\
        "запуска.\n";
const char *g_szArgIsNotFound = "Параметр с ключом -\"%1\" не найден.\n";
const char *g_szFileDoesNotExist = "Файл \"%1\" не существует.\n";
const char *g_szMlpReadingError = "Невозможно загрузить нейронную сеть из "\
        "файла \"%1\".\n";
const char *g_szMlpWritingError = "Невозможно сохранить нейронную сеть в "\
        "файле \"%1\".\n";
const char *g_szTrainsetReadingError = "Невозможно загрузить обучающее "\
        "множество из файла \"%1\".\n";
const char *g_szTrainsetWritingError = "Невозможно сохранить обучающее "\
        "множество в файле \"%1\".\n";
const char *g_szTrainsetStructureError = "Структура обучающего множества не "\
        "соответствует структуре нейронной сети.\n";
const char *g_szControlsetReadingError = "Невозможно загрузить контрольное "\
        "множество из файла \"%1\".\n";
const char *g_szControlsetWritingError = "Невозможно сохранить контрольное "\
        "множество в файле \"%1\".\n";
const char *g_szControlsetStructureError ="Структура контрольного множества не "\
        "соответствует структуре нейронной сети.\n";
const char *g_szDatasetReadingError = "Невозможно загрузить входное "\
        "множество данных из файла \"%1\".\n";
const char *g_szDatasetWritingError =  "Невозможно сохранить входное "\
        "множество данных в файле \"%1\".\n";
const char *g_szDatasetStructureError = "Структура входного множества данных "\
        "не соответствует структуре нейронной сети.\n";
const char *g_szIndexesError = "Индексы весового коэффициента нейронной сети "\
        "заданы неверно.\n";
const char *g_szLayerIndexError = "Индекс слоя задан неверно.\n";
const char *g_szNeuronIndexError = "Индекс нейрона задан неверно.\n";
const char *g_szInputIndexError = "Индекс входа задан неверно.\n";
const char *g_szMlpStructureError = "Данная структура нейронной сети "\
        "некорректна.\n";
const char *g_szIncorrectWeight = "Новое значение весового коэффициента "\
        "некорректно.\n";
const char *g_szUnknownKeyValue = "Значение ключа \"%1\" не распознано.\n";
const char* g_szUnknownTrainingAlgorithm = "Название алгоритма обучения "\
        "неизвестно.\n";
const char* g_szIncorrectMedfiltOrder = "Порядок медианного фильтра задан "\
        "неверно.\n";
const char* g_szMedfiltOrderIsVeryLarge = "Порядок медианного фильтра должен "\
        "быть меньше числа эпох обучения.\n";
const char* g_szIncorrectTheta = "Значение параметра Theta в алгоритме " \
        "Incremental Delta Bar Delta задано неверно.\n";
const char* g_szMaxLearningRateIncorrect = "Верхний предел коэффициента "\
        "скорости обучения задан неверно.\n";
const char* g_szMaxLearningRateItersIncorrect = "Максимальное количество "\
        "итераций алгоритма одномерной оптимизации (подбора оптимального "\
        "коэффициента скорости обучения) задано неверно.\n";
const char* g_szLearningRateIncorrect = "Коэффициент скорости обучения задан "\
        "некорректно.\n";
const char* g_szMaxEpochsIncorrect = "Максимальное число эпох обучения "\
        "задано неверно.\n";
const char* g_szRestartsIncorrect = "Число рестартов алгоритма обучения "\
        "задано неверно.\n";
const char* g_szNullTarget = "Размер вектора желаемого выходного сигнала "\
        "нулевой.\n";
const char* g_szControlsetCannotBeCreated = "Контрольное множество не может "\
        "быть выделено из исходного обучающего множества, поскольку "\
        "количество примеров в исходном множестве слишком мало.";
const char* g_szGradientEpsIncorrect = "Точность алгоритма сопряжённых "\
        "градиентов задана неверно.\n";
const char* g_szSearchEpsIncorrect = "Точность алгоритма одномерной "\
        "оптимизации (подбора оптимального коэффициента скорости обучения) "\
        "задана неверно.\n";
const char* g_szTrainingAlgIncorrect = "Тип алгоритма обучения не "\
        "распознан.\n";
const char* g_szIncorrectTask = "Тип решаемой задачи (классификация или "\
        "регрессия) не распознан.\n";
const char* g_szSeparationFactorIncorrect = "Коэффициент разбиения исходного "\
        "обучающего множества на собственно обучающее и контрольное задан "\
        "некорректно.\n";
const char* g_szIncorrectGoal = "Целевое значение ошибки обучения задано "\
        "неверно.\n";
const char* g_szIncorrectInputsNumber = "Размер входного сигнала задан "\
        "неверно.\n";
const char* g_szIncorrectOutputsNumber = "Размер выходного сигнала задан "\
        "неверно.\n";
const char* g_szCSVReadingError = "Невозможно прочитать данные в формате CSV "\
        "из файла \"%1\".\n";
const char* g_szCSVWritingError = "Невозможно записать данные в формате CSV "\
        "в файл \"%1\".\n";
const char* g_szUnknownDivergentProcessing = "Команда обработки "\
        "противоречивых примеров неизвестна.\n";
const char* g_szDivergentSearchImpossible = "В примерах заданного обучающего "\
        "множества отсутствуют желаемые выходные сигналы, поэтому найти "\
        "противоречивые примеры в этом обучающем множестве невозможно.\n";
