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

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include <QCoreApplication>
#include <QTextCodec>

#include "additional_unit.h"
#include "commands_unit.h"
#include "error_messages.h"
#include "randlib.h"
#include "trainer_unit.h"

using namespace std;

int main(int argc, char *argv[])
{
    setlocale(LC_CTYPE, "Russian");

    QCoreApplication app(argc, argv);
    QTextCodec::setCodecForLocale(QTextCodec::codecForName("System"));

    int nResult = EXIT_SUCCESS;

    try
    {
        initialize_random_generator(time(0));

        TCmdParams cmd_params;

        parse_command_line(argc, argv, cmd_params);
        if (!cmd_params.isEmpty())
        {
            TExecutionMode mode = detect_mode(cmd_params);
            switch (mode)
            {
            case UNKNOWN_MODE:
                fprintf(stderr, g_szIncorrectExecutionMode);
                nResult = EXIT_FAILURE;
                break;
            case SET_MLP_STRUCTURE:
                if (!setStructureOfMLP(cmd_params))
                {
                    nResult = EXIT_FAILURE;
                }
                break;
            case GET_MLP_STRUCTURE:
                if (!getStructureOfMLP(cmd_params))
                {
                    nResult = EXIT_FAILURE;
                }
                break;
            case SET_MLP_WEIGHT:
                if (!setWeightOfMLP(cmd_params))
                {
                    nResult = EXIT_FAILURE;
                }
                break;
            case GET_MLP_WEIGHT:
                if (!getWeightOfMLP(cmd_params))
                {
                    nResult = EXIT_FAILURE;
                }
                break;
            case INITIALIZE_MLP:
                if (!initializeMLP(cmd_params))
                {
                    nResult = EXIT_FAILURE;
                }
                break;
            case TRAIN_MLP:
                {
                    CTrainerForMLP trainer;
                    if (trainer.initialize_params(cmd_params))
                    {
                        if (!trainer.do_training())
                        {
                            nResult = EXIT_FAILURE;
                        }
                    }
                    else
                    {
                        nResult = EXIT_FAILURE;
                    }
                }
                break;
            case USE_MLP:
                if (!useMLP(cmd_params))
                {
                    nResult = EXIT_FAILURE;
                }
                break;
            case SEPARATE_TRAINSET:
                if (!separateTrainset(cmd_params))
                {
                    nResult = EXIT_FAILURE;
                }
                break;
            case SHOW_TRAINSET:
                if (!showTrainset(cmd_params))
                {
                    nResult = EXIT_FAILURE;
                }
                break;
            case CSV_TO_TRAINSET:
                if (!CSVtoTrainset(cmd_params))
                {
                    nResult = EXIT_FAILURE;
                }
                break;
            case TRAINSET_TO_CSV:
                if (!trainsetToCSV(cmd_params))
                {
                    nResult = EXIT_FAILURE;
                }
                break;
            case PROCESS_DIVERGENT_SAMPLES:
                if (!processDivergentTrainSamples(cmd_params))
                {
                    nResult = EXIT_FAILURE;
                }
                break;
            case REMOVE_REPEATING_SAMPLES:
                if (!deleteRepeatingTrainSamples(cmd_params))
                {
                    nResult = EXIT_FAILURE;
                }
                break;
            }
        }
        else
        {
            fprintf(stderr, qPrintable(g_szNoArgs));
            nResult = EXIT_FAILURE;
        }
    }
    catch(exception& e)
    {
        fprintf(stderr, "%s\n", e.what());
        nResult = EXIT_FAILURE;
    }
    catch(...)
    {
        fprintf(stderr, qPrintable(g_szUnknownError));
        nResult = EXIT_FAILURE;
    }

    return nResult;
}
