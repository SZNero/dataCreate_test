#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtGuiTestMotion.h"

class QtGuiTestMotion : public QMainWindow
{
	Q_OBJECT

public:
	QtGuiTestMotion(QWidget *parent = Q_NULLPTR);

private:
	Ui::QtGuiTestMotionClass ui;
};
