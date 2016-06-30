#include <cmath>
#include <ctime>
#include <vector>
#include <QtTest/QtTest>
#include "../annlib.h"
#include "../randlib.h"

const double g_accuracy = 0.000001;

class Test_ANNLib: public QObject
{
    Q_OBJECT
private:
    CMultilayerPerceptron m_mlp_i5_4lin_4sig_3max;
    CMultilayerPerceptron m_mlp_i5_4lin_4sig_3lin;
    int m_nSamplesNumber;
    std::vector<double> m_aInputs;
    std::vector<double> m_aTargets, m_aBinaryTargets;
    std::vector<double> m_aOutputs, m_aBinaryOutputs, m_aLinearOutputs;
    std::vector<double> m_aWeightsOfSamples;
    double m_MSE, m_weightedMSE;
    double m_CrEnt, m_weightedCrEnt;
    double m_ClassErr, m_weightedClassErr;
    double m_RegrErr, m_weightedRegrErr;
private slots:
    void initTestCase();
    void calculate_outputs();
    void calculate_mse();
    void calculate_weighted_mse();
    void calculate_error();
    void calculate_weighted_error();
    void initialize_weights();
};

void Test_ANNLib::initTestCase()
{
    const int aSizesOfLayers[] = {4, 4, 3};
    const TActivationKind aActivationFunctions[] = {LIN, SIG, SOFT};
    m_mlp_i5_4lin_4sig_3max.resize(5, 3, aSizesOfLayers, aActivationFunctions);

    // Веса 1-го нейрона 1-го слоя (последний вес - это смещение)
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 0, 0, 0.220544867845439);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 0, 1, -0.463038696912156);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 0, 2, 0.321655107405743);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 0, 3, -0.291469137048367);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 0, 4, -0.111219555959633);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 0, 5, 0.226148077692062);

    // Веса 2-го нейрона 1-го слоя (последний вес - это смещение)
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 1, 0, -0.0533482980168669);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 1, 1, 0.0538419201301921);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 1, 2, -0.416571468199462);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 1, 3, -0.498299005505861);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 1, 4, 0.213229398849361);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 1, 5, -0.164472726074836);

    // Веса 3-го нейрона 1-го слоя (последний вес - это смещение)
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 2, 0, -0.182864643668493);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 2, 1, 0.487115988242971);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 2, 2, -0.428907288659582);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 2, 3, -0.180696112233001);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 2, 4, 0.151300590550566);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 2, 5, 0.356827236958567);

    // Веса 4-го нейрона 1-го слоя (последний вес - это смещение)
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 3, 0, 0.140134387309285);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 3, 1, 0.414220490719553);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 3, 2, -0.234815275264621);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 3, 3, 0.445042938043802);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 3, 4, -0.304534628112927);
    m_mlp_i5_4lin_4sig_3max.setWeight(0, 3, 5, 0.215333713025894);

    // Веса 1-го нейрона 2-го слоя (последний вес - это смещение)
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 0, 0, -0.306544558093031);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 0, 1, 0.384380309884593);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 0, 2, 0.411915060004995);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 0, 3, -0.171582412570231);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 0, 4, -0.4695369777383);

    // Веса 2-го нейрона 2-го слоя (последний вес - это смещение)
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 1, 0, -0.127581949800445);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 1, 1, 0.276825370444168);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 1, 2, 0.213120337853734);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 1, 3, -0.0504241103314687);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 1, 4, 0.0496940865679223);

    // Веса 3-го нейрона 2-го слоя (последний вес - это смещение)
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 2, 0, -0.466535641552545);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 2, 1, -0.157522514610854);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 2, 2, 0.474207004585998);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 2, 3, 0.462384459082861);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 2, 4, 0.042378964388474);

    // Веса 4-го нейрона 2-го слоя (последний вес - это смещение)
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 3, 0, -0.341253180621593);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 3, 1, 0.349933235941731);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 3, 2, -0.041903329006605);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 3, 3, 0.0821481299158505);
    m_mlp_i5_4lin_4sig_3max.setWeight(1, 3, 4, -0.0142072190530919);

    // Веса 1-го нейрона 3-го слоя (последний вес - это смещение)
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 0, 0, 0.0333305811793485);
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 0, 1, -0.242963534499817);
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 0, 2, -0.0663860683690252);
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 0, 3, 0.450021463349826);
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 0, 4, -0.387454075669828);

    // Веса 2-го нейрона 3-го слоя (последний вес - это смещение)
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 1, 0, -0.0459452010365927);
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 1, 1, -0.152378265578448);
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 1, 2, 0.396466634760342);
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 1, 3, 0.242677298355983);
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 1, 4, -0.308444579477127);

    // Веса 3-го нейрона 3-го слоя (последний вес - это смещение)
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 2, 0, 0.132175055547954);
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 2, 1, -0.121483564385766);
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 2, 2, 0.11043702371558);
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 2, 3, 0.237671958485141);
    m_mlp_i5_4lin_4sig_3max.setWeight(2, 2, 4, -0.200111055995077);

    m_mlp_i5_4lin_4sig_3lin = m_mlp_i5_4lin_4sig_3max;
    m_mlp_i5_4lin_4sig_3lin.setActivationKind(2, LIN);

    m_nSamplesNumber = 3;
    m_aInputs.resize(m_nSamplesNumber
                     * m_mlp_i5_4lin_4sig_3max.getInputsCount());
    m_aTargets.resize(m_nSamplesNumber * aSizesOfLayers[2]);
    m_aBinaryTargets.resize(m_aTargets.size());
    m_aOutputs.resize(m_aTargets.size());
    m_aBinaryOutputs.resize(m_aTargets.size());
    m_aLinearOutputs.resize(m_aTargets.size());
    m_aWeightsOfSamples.resize(m_nSamplesNumber);
    const double aInputs[] = {
        0.8731258121, 0.3028962075, 0.5347624655, 0.6449618499, 0.5626002472,
        0.2910496267, 0.7649372026, 0.6217597046, 0.06925209802, 0.8529230425,
        0.6439631525, 0.6416425724, 0.439495312, 0.4182047141, 0.8023272511
    };
    const double aLinearOutputs[] = {
        -0.5722019034, -0.1476912832, -0.3357051518,
        -0.5735650656, -0.1234177009, -0.2765729879,
        -0.56370728, -0.07749239268, -0.276277901
    };
    const double aOutputs[] = {
        0.2634597994, 0.4027884203, 0.3337517803,
        0.2554703904, 0.4007163734, 0.3438132362,
        0.2525797327, 0.4107323801, 0.3366878873
    };
    const double aBinaryOutputs[] = {
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0
    };
    const double aTargets[] = {
        0.06493548089, 0.2162486581, -0.1292967022,
        -0.4823101686, 0.3538098262, 0.01411220996,
        -0.4439190059, -0.2457882757, -0.1747573881
    };
    const double aBinaryTargets[] = {
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    const double aWeightsOfSamples[] = {0.2, 0.3, 0.5};
    int i, j, k;
    const int nOutputSize = aSizesOfLayers[2];
    for (i = 0; i < m_nSamplesNumber; i++)
    {
        for (j = 0; j < m_mlp_i5_4lin_4sig_3max.getInputsCount(); j++)
        {
            k = i * m_mlp_i5_4lin_4sig_3max.getInputsCount() + j;
            m_aInputs[k] = aInputs[k];
        }
        for (j = 0; j < nOutputSize; j++)
        {
            k = i * nOutputSize + j;
            m_aTargets[k] = aTargets[k];
            m_aBinaryTargets[k] = aBinaryTargets[k];
            m_aLinearOutputs[k] = aLinearOutputs[k];
            m_aOutputs[k] = aOutputs[k];
            m_aBinaryOutputs[k] = aBinaryOutputs[k];
        }
        m_aWeightsOfSamples[i] = aWeightsOfSamples[i];
    }

    m_MSE = 0.106061264210;
    m_weightedMSE = 0.079620388568;

    m_CrEnt = 0.323604910554;
    m_weightedCrEnt = 0.333506219169;

    m_ClassErr = 33.333333333333;
    m_weightedClassErr = 50.0;

    m_RegrErr = 441.032172526631;
    m_weightedRegrErr = 419.702834425278;

    initialize_random_generator(time(0));
}

void Test_ANNLib::calculate_outputs()
{
    const char *szErrMsg = "calculated %1 != target %2";
    std::vector<double> outputs(m_aOutputs.size());

    m_mlp_i5_4lin_4sig_3max.calculate_outputs(
                &m_aInputs[0], &outputs[0], m_nSamplesNumber);
    for (unsigned i = 0; i < outputs.size(); i++)
    {
        QVERIFY2(fabs(outputs[i] - m_aOutputs[i]) <= g_accuracy,
                 QString(szErrMsg).arg(outputs[i]).arg(
                     m_aOutputs[i]).toStdString().c_str());
    }

    m_mlp_i5_4lin_4sig_3lin.calculate_outputs(
                &m_aInputs[0], &outputs[0], m_nSamplesNumber);
    for (unsigned i = 0; i < outputs.size(); i++)
    {
        QVERIFY2(fabs(outputs[i] - m_aLinearOutputs[i]) <= g_accuracy,
                 QString(szErrMsg).arg(outputs[i]).arg(
                     m_aLinearOutputs[i]).toStdString().c_str());
    }
}

void Test_ANNLib::calculate_mse()
{
    const char *szErrMsg1 = "Cross entropy: calculated %1 != target %2";
    const char *szErrMsg2 = "MSE: calculated %1 != target %2";

    double calculated = m_mlp_i5_4lin_4sig_3max.calculate_mse(
                &m_aInputs[0], &m_aBinaryTargets[0], m_nSamplesNumber);
    QVERIFY2(fabs(calculated - m_CrEnt) <= g_accuracy,
             QString(szErrMsg1).arg(calculated).arg(
                 m_CrEnt).toStdString().c_str());

    calculated = m_mlp_i5_4lin_4sig_3lin.calculate_mse(
                &m_aInputs[0], &m_aTargets[0], m_nSamplesNumber);
    QVERIFY2(fabs(calculated - m_MSE) <= g_accuracy,
             QString(szErrMsg2).arg(calculated).arg(
                 m_MSE).toStdString().c_str());
}

void Test_ANNLib::calculate_weighted_mse()
{
    const char *szErrMsg1 = "Cross entropy: calculated %1 != target %2";
    const char *szErrMsg2 = "MSE: calculated %1 != target %2";

    double calculated = m_mlp_i5_4lin_4sig_3max.calculate_mse(
                &m_aInputs[0], &m_aBinaryTargets[0], &m_aWeightsOfSamples[0],
                m_nSamplesNumber);
    QVERIFY2(fabs(calculated - m_weightedCrEnt) <= g_accuracy,
             QString(szErrMsg1).arg(calculated).arg(
                 m_weightedCrEnt).toStdString().c_str());

    calculated = m_mlp_i5_4lin_4sig_3lin.calculate_mse(
                &m_aInputs[0], &m_aTargets[0], &m_aWeightsOfSamples[0],
                m_nSamplesNumber);
    QVERIFY2(fabs(calculated - m_weightedMSE) <= g_accuracy,
             QString(szErrMsg2).arg(calculated).arg(
                 m_weightedMSE).toStdString().c_str());
}

void Test_ANNLib::calculate_error()
{
    const char *szErrMsg1 = "Classification: calculated %1 != target %2";
    const char *szErrMsg2 = "Regression: calculated %1 != target %2";

    double calculated = m_mlp_i5_4lin_4sig_3max.calculate_error(
                &m_aInputs[0], &m_aBinaryTargets[0], m_nSamplesNumber,
                taskCLASSIFICATION);
    QVERIFY2(fabs(calculated - m_ClassErr) <= g_accuracy,
             QString(szErrMsg1).arg(calculated).arg(
                 m_ClassErr).toStdString().c_str());

    calculated = m_mlp_i5_4lin_4sig_3max.calculate_error(
                &m_aInputs[0], &m_aTargets[0], m_nSamplesNumber,
                taskREGRESSION);
    QVERIFY2(fabs(calculated - m_RegrErr) <= g_accuracy,
             QString(szErrMsg2).arg(calculated).arg(
                 m_RegrErr).toStdString().c_str());
}

void Test_ANNLib::calculate_weighted_error()
{
    const char *szErrMsg1 = "Classification: calculated %1 != target %2";
    const char *szErrMsg2 = "Regression: calculated %1 != target %2";

    double calculated = m_mlp_i5_4lin_4sig_3max.calculate_error(
                &m_aInputs[0], &m_aBinaryTargets[0], &m_aWeightsOfSamples[0],
                m_nSamplesNumber, taskCLASSIFICATION);
    QVERIFY2(fabs(calculated - m_weightedClassErr) <= g_accuracy,
             QString(szErrMsg1).arg(calculated).arg(
                 m_weightedClassErr).toStdString().c_str());

    calculated = m_mlp_i5_4lin_4sig_3max.calculate_error(
                &m_aInputs[0], &m_aTargets[0], &m_aWeightsOfSamples[0],
                m_nSamplesNumber, taskREGRESSION);
    QVERIFY2(fabs(calculated - m_weightedRegrErr) <= g_accuracy,
             QString(szErrMsg2).arg(calculated).arg(
                 m_weightedRegrErr).toStdString().c_str());
}

void Test_ANNLib::initialize_weights()
{
    int iLayer, iNeuron, iWeight;
    double new_weight, old_weight;
    bool equal, zero;
    int nInputsNumber = m_mlp_i5_4lin_4sig_3max.getInputsCount();
    CMultilayerPerceptron new_mlp = m_mlp_i5_4lin_4sig_3max;
    new_mlp.initialize_weights();
    QCOMPARE(new_mlp.getInputsCount(),
             m_mlp_i5_4lin_4sig_3max.getInputsCount());
    QCOMPARE(new_mlp.getLayersCount(),
             m_mlp_i5_4lin_4sig_3max.getLayersCount());
    for (iLayer = 0; iLayer < new_mlp.getLayersCount(); iLayer++)
    {
        QCOMPARE(new_mlp.getLayerSize(iLayer),
                 m_mlp_i5_4lin_4sig_3max.getLayerSize(iLayer));
        QCOMPARE(new_mlp.getActivationKind(iLayer),
                 m_mlp_i5_4lin_4sig_3max.getActivationKind(iLayer));
        for (iNeuron = 0; iNeuron < new_mlp.getLayerSize(iLayer); iNeuron++)
        {
            equal = true;
            zero = true;
            for (iWeight = 0; iWeight <= nInputsNumber; iWeight++)
            {
                old_weight = m_mlp_i5_4lin_4sig_3max.getWeight(iLayer, iNeuron,
                                                               iWeight);
                new_weight = new_mlp.getWeight(iLayer, iNeuron, iWeight);
                if (fabs(new_weight - old_weight) > g_accuracy)
                {
                    equal = false;
                }
                if (fabs(new_weight) > g_accuracy)
                {
                    zero = false;
                }
            }
            QVERIFY(!equal);
            QVERIFY(!zero);
        }
        nInputsNumber = new_mlp.getLayerSize(iLayer);
    }
}

QTEST_MAIN(Test_ANNLib)
#include "test.moc"
