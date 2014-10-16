# -------------------------------------------------
# Project created by QtCreator 2010-03-21T15:57:32
# -------------------------------------------------
QT -= gui
QT += core
TARGET = nnsys
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = app
SOURCES += main.cpp \
    additional_unit.cpp \
    annlib.cpp \
    commands_unit.cpp \
    error_messages.cpp \
    trainer_unit.cpp \
    randlib.cpp \
    mathlib_bond005.cpp
HEADERS += additional_unit.h \
    annlib.h \
    commands_unit.h \
    error_messages.h \
    trainer_unit.h \
    randlib.h \
    mathlib_bond005.h
win32:QMAKE_CXXFLAGS += /fp:fast /Ox /arch:AVX /openmp
unix:QMAKE_CXXFLAGS += -O3 -march=native -mfpmath=sse -msse2 -funroll-loops -ffast-math -fopenmp
unix:QMAKE_LIBS+= -lgomp -lpthread
