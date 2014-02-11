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

#ifndef ADDITIONAL_UNIT_H
#define ADDITIONAL_UNIT_H

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

#include <QMap>
#include <QString>

class EIncorrectArg: public std::exception
{
private:
    QString m_sErrorMsg;
public:
    EIncorrectArg(int iArg) throw();
    virtual ~EIncorrectArg() throw() {};
    virtual const char* what() const throw();
};

typedef QMap<QString, QString> TCmdParams;

/* Сравнение строк szText1 и szText2 без учёта регистра символов. Если строки
равны, то вовзращается true, если же не равны - false. */
bool same_text_nocase(const char *szText1, const char *szText2);

/* Преобразование строки sSrc в вещественное число rDst типа float. В случае
успешного преобразования возвращается true, в случае ошибки - false. */
bool string_to_real(const std::string& sSrc, float& rDst);

/* Преобразование строки sSrc в вещественное число rDst типа double. В случае
успешного преобразования возвращается true, в случае ошибки - false. */
bool string_to_real(const std::string& sSrc, double& rDst);

/* Преобразование строки sSrc в вещественное число rDst типа long double.
В случае успешного преобразования возвращается true, в случае ошибки - false.*/
bool string_to_real(const std::string& sSrc, long double& rDst);

/* Преобразование строки sSrc в целое число rDst. В случае успешного
преобразования возвращается true, в случае ошибки - false. */
bool string_to_integer(const std::string& sSrc, int& rDst);

/* Все аргументы командной строки должны иметь вид:
   -<строка-ключ>=<строка-значение>
   или
   -<строка-ключ>
   Данная функция выполняет анализ аргументов командной строки на предмет
соответствия этим шаблонам и добавляет найденные пары "ключ-значение" в словарь
параметров rParams, предварительное преобразовав символы ключа к нижнему
регистру. Кроме того, ни одна строка-ключ не должна повторяться два или более
раз.
   В случае, если вышеописанные условия выполняются и словарь параметров
успешно сформирован, функция срабатывает нормально. Также нормальное срабатывание
происходит, если аргументы командной строки отсутствуют. Остальные случаи
расцениваются как ошибки, и тогда возвращается возбуждается исключение
EIncorrectArg. */
void parse_command_line(int argc, char *argv[], TCmdParams& rParams);

/* Распечатать (вывести в stdout) строку sLine. Если её длина меньше width, то
дополнить справа пробелами. */
void print_line(const QString& sLine, int width = 0);

/* Удалить пробельные символы и символы табуляции из начала и из конца строки
sLine. */
void trim_line(QString& sLine);

/* Удалить пробельные символы и символы табуляции из начала и из конца строки
sLine. */
void trim_line(std::string& sLine);

/* Проверить, действительно ли символ c является пробельным символом. */
inline bool is_space(char c)
{
    return ((c == ' ') || (c <= 13));
};

/* Проверить, действительно ли символ c является пробельным символом. */
inline bool is_space(QChar c)
{
    return ((c == ' ') || (c <= 13));
};


#endif // ADDITIONAL_UNIT_H
