/********************************************************************************
** Form generated from reading UI file 'QtGuiTestMotion.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QTGUITESTMOTION_H
#define UI_QTGUITESTMOTION_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QtGuiTestMotionClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *QtGuiTestMotionClass)
    {
        if (QtGuiTestMotionClass->objectName().isEmpty())
            QtGuiTestMotionClass->setObjectName(QStringLiteral("QtGuiTestMotionClass"));
        QtGuiTestMotionClass->resize(600, 400);
        menuBar = new QMenuBar(QtGuiTestMotionClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        QtGuiTestMotionClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(QtGuiTestMotionClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        QtGuiTestMotionClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(QtGuiTestMotionClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        QtGuiTestMotionClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(QtGuiTestMotionClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        QtGuiTestMotionClass->setStatusBar(statusBar);

        retranslateUi(QtGuiTestMotionClass);

        QMetaObject::connectSlotsByName(QtGuiTestMotionClass);
    } // setupUi

    void retranslateUi(QMainWindow *QtGuiTestMotionClass)
    {
        QtGuiTestMotionClass->setWindowTitle(QApplication::translate("QtGuiTestMotionClass", "QtGuiTestMotion", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class QtGuiTestMotionClass: public Ui_QtGuiTestMotionClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTGUITESTMOTION_H
