QT += core testlib
QT -= gui

CONFIG += c++11

TARGET = annlib_tests
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += \
    test.cpp \
    ../annlib.cpp \
    ../mathlib_bond005.cpp \
    ../randlib.cpp

HEADERS += \
    ../annlib.h \
    ../mathlib_bond005.h \
    ../randlib.h
