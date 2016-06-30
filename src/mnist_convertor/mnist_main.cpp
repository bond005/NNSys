#include <cstdlib>
#include <iostream>

#include <QDir>
#include <QString>
#include <QVector>

#include <QFile>
#include <QDataStream>
#include <QTextStream>

static const int g_nSizeOfTrainSet = 60000;
static const int g_nSizeOfTestSet = 10000;

static const qint32 g_nMagicNumberForInputs = 2051;
static const qint32 g_nMagicNumberForTargets = 2049;
static const qint32 g_nDesiredImageRows = 28;
static const qint32 g_nDesiredImageCols = 28;

static const qint32 g_nInputsNumber = 28 * 28;
static const qint32 g_nTargetsNumber = 10;

static const char* g_szTrainInputsName = "train-images.idx3-ubyte";
static const char* g_szTrainTargetsName = "train-labels.idx1-ubyte";
static const char* g_szTestInputsName = "t10k-images.idx3-ubyte";
static const char* g_szTestTargetsName = "t10k-labels.idx1-ubyte";

static const char* g_szTrainSetName = "mnist_trainset.dat";
static const char* g_szTrainSetNameWithoutTargets = "mnist_trainset_without_targets.dat";
static const char* g_szTestSetName = "mnist_testset.dat";
static const char* g_szTestSetNameWithoutTargets = "mnist_testset_without_targets.dat";
static const char* g_szNameOfTrainLabels = "mnist_trainset_labels.txt";
static const char* g_szNameOfTestLabels = "mnist_testset_labels.txt";

static const char* g_szNamesOfClasses[10] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
};

bool loadInputs(const QString& sFileName, const qint32 nSamplesNumber,
                QVector<double>& aInputs);
bool loadTargets(const QString& sFileName, const qint32 nSamplesNumber,
                 QVector<double>& aTargets, QVector<QString>& aLabelsOfClasses);
bool saveTrainSet(const QString& sFileName, const double aTrainInputs[],
                  const double aTrainTargets[], qint32 nSamplesNumber);
bool saveLabelsOfClasses(const QString& sFileName,
                         QVector<QString>& aLabelsOfClasses);

int main(int argc, char *argv[])
{
    if ((argc != 3) && (argc != 4))
    {
        if (argc < 3)
        {
            std::cerr << "There are too few parameters of the command prompt."
                      << std::endl;
        }
        else
        {
            std::cerr << "There are too many parameters of the command prompt."
                      << std::endl;
        }
        return EXIT_FAILURE;
    }

    QDir srcDir(argv[1]), dstDir(argv[2]);
    if (!srcDir.exists())
    {
        std::cerr << "The source directory doesn't exist." << std::endl;
        return EXIT_FAILURE;
    }
    if (!dstDir.exists())
    {
        std::cerr << "The destination directory doesn't exist." << std::endl;
        return EXIT_FAILURE;
    }
    bool bIsInputsOnly = false;
    if (argc == 4)
    {
        if (QString(argv[3]).compare("-inputs_only") != 0)
        {
            std::cerr << "There are too many parameters of the command prompt."
                      << std::endl;
            return EXIT_FAILURE;
        }
        bIsInputsOnly = true;
    }

    QVector<double> aTrainInputs, aTrainTargets, aTestInputs, aTestTargets;
    QVector<QString> aTrainLabels, aTestLabels;

    if (!loadInputs(srcDir.absoluteFilePath(g_szTrainInputsName),
                    g_nSizeOfTrainSet, aTrainInputs))
    {
        std::cerr << "The file with train images cannot be loaded."
                  << std::endl;
        return EXIT_FAILURE;
    }
    if (!loadTargets(srcDir.absoluteFilePath(g_szTrainTargetsName),
                     g_nSizeOfTrainSet, aTrainTargets, aTrainLabels))
    {
        std::cerr << "The file with train labels cannot be loaded."
                  << std::endl;
        return EXIT_FAILURE;
    }
    if (!loadInputs(srcDir.absoluteFilePath(g_szTestInputsName),
                    g_nSizeOfTestSet, aTestInputs))
    {
        std::cerr << "The file with test images cannot be loaded."
                  << std::endl;
        return EXIT_FAILURE;
    }
    if (!loadTargets(srcDir.absoluteFilePath(g_szTestTargetsName),
                     g_nSizeOfTestSet, aTestTargets, aTestLabels))
    {
        std::cerr << "The file with test labels cannot be loaded."
                  << std::endl;
        return EXIT_FAILURE;
    }

    if (!saveTrainSet(dstDir.absoluteFilePath(
                          bIsInputsOnly ? g_szTrainSetNameWithoutTargets
                          : g_szTrainSetName),
                      &aTrainInputs[0], bIsInputsOnly ? 0 : &aTrainTargets[0],
                      g_nSizeOfTrainSet))
    {
        std::cerr << "The created train set cannot be saved." << std::endl;
        return EXIT_FAILURE;
    }
    if (!saveLabelsOfClasses(dstDir.absoluteFilePath(g_szNameOfTrainLabels),
                             aTrainLabels))
    {
        std::cerr << "The classes labels for the train set cannot be saved."
                  << std::endl;
        return EXIT_FAILURE;
    }
    if (!saveTrainSet(dstDir.absoluteFilePath(
                          bIsInputsOnly ? g_szTestSetNameWithoutTargets
                          : g_szTestSetName),
                      &aTestInputs[0], bIsInputsOnly ? 0 : &aTestTargets[0],
                      g_nSizeOfTestSet))
    {
        std::cerr << "The created test set cannot be saved." << std::endl;
        return EXIT_FAILURE;
    }
    if (!saveLabelsOfClasses(dstDir.absoluteFilePath(g_szNameOfTestLabels),
                             aTestLabels))
    {
        std::cerr << "The classes labels for the test set cannot be saved."
                  << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

bool loadInputs(const QString& sFileName, const qint32 nSamplesNumber,
                QVector<double>& aInputs)
{
    QFile inputsFile(sFileName);
    if (!inputsFile.open(QIODevice::ReadOnly))
    {
        return false;
    }

    QDataStream inputsStream(&inputsFile);
    inputsStream.setByteOrder(QDataStream::BigEndian);
    if (inputsStream.status() != QDataStream::Ok)
    {
        return false;
    }

    qint32 nMagicNumber = 0, nImagesCount = 0;
    qint32 nImageRows = 0, nImageCols = 0;

    inputsStream >> nMagicNumber >> nImagesCount >> nImageRows >> nImageCols;
    if (inputsStream.status() != QDataStream::Ok)
    {
        return false;
    }
    if (nMagicNumber != g_nMagicNumberForInputs)
    {
        return false;
    }
    if (nImagesCount <= 0)
    {
        return false;
    }
    if ((nImageRows != g_nDesiredImageRows)
            || (nImageCols != g_nDesiredImageCols)
            || (nImagesCount != nSamplesNumber))
    {
        return false;
    }

    bool ok = true;
    int iInput = 0;
    qint32 i, j;
    quint8 value;

    aInputs.resize(nSamplesNumber * g_nInputsNumber);
    for (i = 0; i < nSamplesNumber; i++)
    {
        for (j = 0; j < g_nInputsNumber; j++)
        {
            inputsStream >> value;
            if (inputsStream.status() != QDataStream::Ok)
            {
                ok = false;
                break;
            }
            aInputs[iInput++] = (double)value / 255.0;
        }
        if (!ok)
        {
            break;
        }
    }

    return ok;
}

bool loadTargets(const QString& sFileName, const qint32 nSamplesNumber,
                 QVector<double>& aTargets, QVector<QString>& aLabelsOfClasses)
{
    QFile targetsFile(sFileName);
    if (!targetsFile.open(QIODevice::ReadOnly))
    {
        return false;
    }

    QDataStream targetsStream(&targetsFile);
    targetsStream.setByteOrder(QDataStream::BigEndian);
    if (targetsStream.status() != QDataStream::Ok)
    {
        return false;
    }

    qint32 nMagicNumber = 0, nLabelsCount = 0;

    targetsStream >> nMagicNumber >> nLabelsCount;
    if (targetsStream.status() != QDataStream::Ok)
    {
        return false;
    }
    if (nMagicNumber != g_nMagicNumberForTargets)
    {
        return false;
    }
    if (nLabelsCount != nSamplesNumber)
    {
        return false;
    }

    qint32 i, iTarget = 0;
    quint8 label;
    bool ok = true;

    aLabelsOfClasses.clear();
    aLabelsOfClasses.reserve(nSamplesNumber);
    aTargets.resize(nSamplesNumber * g_nTargetsNumber);
    aTargets.fill(0.0);
    for (i = 0; i < nSamplesNumber; i++)
    {
        targetsStream >> label;
        if (targetsStream.status() != QDataStream::Ok)
        {
            ok = false;
            break;
        }
        if (label > 10)
        {
            ok = false;
            break;
        }
        aTargets[iTarget + label] = 1.0;
        aLabelsOfClasses.append(g_szNamesOfClasses[label]);
        iTarget += g_nTargetsNumber;
    }

    return ok;
}

bool saveTrainSet(const QString& sFileName, const double aTrainInputs[],
                  const double aTrainTargets[], qint32 nSamplesNumber)
{
    QFile trainsetFile(sFileName);
    if (!trainsetFile.open(QFile::WriteOnly | QFile::Truncate))
    {
        return false;
    }

    QDataStream trainsetStream(&trainsetFile);
    trainsetStream.setByteOrder(QDataStream::BigEndian);
    trainsetStream.setFloatingPointPrecision(QDataStream::DoublePrecision);
    if (trainsetStream.status() != QDataStream::Ok)
    {
        return false;
    }

    trainsetStream << nSamplesNumber << g_nInputsNumber
                   << ((aTrainTargets == 0) ? 0 : g_nTargetsNumber);
    if (trainsetStream.status() != QDataStream::Ok)
    {
        return false;
    }

    bool ok = true;
    qint32 i, j;
    for (i = 0; i < nSamplesNumber; i++)
    {
        for (j = 0; j < g_nInputsNumber; j++)
        {
            trainsetStream << aTrainInputs[i * g_nInputsNumber + j];
            if (trainsetStream.status() != QDataStream::Ok)
            {
                ok = false;
                break;
            }
        }
        if (!ok)
        {
            break;
        }
        if (aTrainTargets == 0)
        {
            continue;
        }
        for (j = 0; j < g_nTargetsNumber; j++)
        {
            trainsetStream << aTrainTargets[i * g_nTargetsNumber + j];
            if (trainsetStream.status() != QDataStream::Ok)
            {
                ok = false;
                break;
            }
        }
        if (!ok)
        {
            break;
        }
    }

    return ok;
}

bool saveLabelsOfClasses(const QString& sFileName,
                         QVector<QString>& aLabelsOfClasses)
{
    QFile labelsFile(sFileName);
    if (!labelsFile.open(QFile::WriteOnly | QFile::Truncate))
    {
        return false;
    }

    QTextStream labelsStream(&labelsFile);
    if (labelsStream.status() != QTextStream::Ok)
    {
        return false;
    }

    bool ok = true;
    for (int i = 0; i < aLabelsOfClasses.size(); i++)
    {
        labelsStream << aLabelsOfClasses[i] << endl;
        if (labelsStream.status() != QTextStream::Ok)
        {
            ok = false;
            break;
        }
    }

    return ok;
}
