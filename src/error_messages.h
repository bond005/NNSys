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

#ifndef ERROR_MESSAGES_H
#define ERROR_MESSAGES_H

extern const char *g_szUnknownError;
extern const char *g_szNoArgs;
extern const char *g_szFewArgs;
extern const char *g_szManyArgs;
extern const char *g_szImpossibleVal;
extern const char *g_szNullVal;
extern const char *g_szSuperfluousArg;
extern const char *g_szIncorrectExecutionMode;
extern const char *g_szArgIsNotFound;
extern const char *g_szFileDoesNotExist;
extern const char *g_szMlpReadingError;
extern const char *g_szMlpWritingError;
extern const char *g_szTrainsetReadingError;
extern const char *g_szTrainsetWritingError;
extern const char* g_szControlsetWritingError;
extern const char *g_szTrainsetStructureError;
extern const char *g_szControlsetReadingError;
extern const char *g_szControlsetStructureError;
extern const char *g_szControlsetCannotBeCreated;
extern const char *g_szDatasetReadingError;
extern const char *g_szDatasetWritingError;
extern const char *g_szDatasetStructureError;
extern const char *g_szIndexesError;
extern const char *g_szLayerIndexError;
extern const char *g_szNeuronIndexError;
extern const char *g_szInputIndexError;
extern const char *g_szMlpStructureError;
extern const char *g_szIncorrectWeight;
extern const char *g_szUnknownKeyValue;
extern const char *g_szUnknownTrainingAlgorithm;
extern const char *g_szIncorrectMedfiltOrder;
extern const char *g_szMedfiltOrderIsVeryLarge;
extern const char *g_szIncorrectTheta;
extern const char *g_szMaxLearningRateIncorrect;
extern const char* g_szMaxLearningRateItersIncorrect;
extern const char* g_szLearningRateIncorrect;
extern const char *g_szMaxEpochsIncorrect;
extern const char* g_szRestartsIncorrect;
extern const char* g_szNullTarget;
extern const char* g_szControlsetCannotBeCreated;
extern const char* g_szGradientEpsIncorrect;
extern const char* g_szSearchEpsIncorrect;
extern const char* g_szTrainingAlgIncorrect;
extern const char* g_szIncorrectTask;
extern const char* g_szSeparationFactorIncorrect;
extern const char* g_szIncorrectGoal;
extern const char* g_szIncorrectInputsNumber;
extern const char* g_szIncorrectOutputsNumber;
extern const char* g_szCSVReadingError;

#endif // ERROR_MESSAGES_H
