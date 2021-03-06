# -*- coding: utf-8 -*-
"""08-01-introduction-to-sql.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tqRupqNp4WRv-bwgtdNbDHRroDNgSIqC

# Introduction to SQL
## 1. Selecting columns
## 2. Filtering rows
## 3. Aggregate Functions
## 4. Sorting and grouping

## 1. Selecting columns

**Onboarding | Tables**

If you've used DataCamp to learn R or Python, you'll be familiar with the interface. For SQL, however, there are a few new features you should be aware of.

For this course, you'll be using a database containing information on almost 5000 films. To the right, underneath the editor, you can see the data in this database by clicking through the tabs.

From looking at the tabs, who is the first person listed in the people table? Refer to films.sql

**Possible Answers**

- [ ] Kanye West
- [ ] Biggie Smalls
- [x] 50 Cent
- [ ] Jay Z

**Onboarding | Query Result**

Notice the query result tab in the bottom right corner of your screen. This is where the results of your SQL queries will be displayed.

Run this query in the editor and check out the resulting table in the query result tab!

`SELECT name FROM people;`

Who is the second person listed in the query result?

**Possible Answers**

- [ ] Kanye West
- [x] A. Michael Baldwin
- [ ] 50 Cent
- [ ] Jay Z

**Onboarding | Errors**

```
-- Try running me!
SELECT 'DataCamp <3 SQL'
AS result;
```

**Onboarding | Bullet Exercises**

```
SELECT 'SQL'
AS result;

SELECT 'SQL is'
AS result;

SELECT 'SQL is cool'
AS result;
```

**Beginning your SQL journey**

Now that you're familiar with the interface, let's get straight into it.

SQL, which stands for Structured Query Language, is a language for interacting with data stored in something called a relational database.

You can think of a relational database as a collection of tables. A table is just a set of rows and columns, like a spreadsheet, which represents exactly one type of entity. For example, a table might represent employees in a company or purchases made, but not both.

Each row, or record, of a table contains information about a single entity. For example, in a table representing employees, each row represents a single person. Each column, or field, of a table contains a single attribute for all rows in the table. For example, in a table representing employees, we might have a column containing first and last names for all employees.

The table of employees might look something like this:

```
id	name	age	nationality
1	Jessica	22	Ireland
2	Gabriel	48	France
3	Laura	36	USA
```

How many fields does the employees table above contain?

**Possible Answers**

- [ ] 1
- [ ] 2
- [ ] 3
- [x] 4

**SELECTing single columns**

```
SELECT title
FROM films;

SELECT release_year
FROM films;

SELECT name
FROM people;
```

**SELECTing multiple columns**

```
SELECT title
FROM films;

SELECT title, release_year
FROM films;

SELECT title, release_year, country
FROM films;

SELECT *
FROM films;
```

**SELECT DISTINCT**

```
SELECT DISTINCT country
FROM films;

SELECT DISTINCT certification
FROM films;

SELECT DISTINCT role
FROM roles;
```

**Learning to COUNT**

What if you want to count the number of employees in your employees table? The COUNT statement lets you do this by returning the number of rows in one or more columns.

For example, this code gives the number of rows in the people table:

```
SELECT COUNT(*)
FROM people;
```

How many records are contained in the reviews table?

**Possible Answers**

- [ ] 9,468
- [ ] 8,397
- [x] 4,968
- [ ] 9,837
- [ ] 9,864

**Practice with COUNT**

```
SELECT COUNT(*)
FROM people;

SELECT COUNT(birthdate)
FROM people;

SELECT COUNT(DISTINCT birthdate)
FROM people;

SELECT COUNT(DISTINCT language)
FROM films;

SELECT COUNT(DISTINCT country)
FROM films;
```

## 2. Filtering rows

**Filtering results**

In SQL, the WHERE keyword allows you to filter based on both text and numeric values in a table. 

There are a few different comparison operators you can use:

- `= equal`
- `<> not equal`
- `< less than`
- `> greater than`
- `<= less than or equal to`
- `>= greater than or equal to`

For example, you can filter text records such as title. The following code returns all films with the title 'Metropolis':

```
SELECT title
FROM films
WHERE title = 'Metropolis';
```

Notice that the WHERE clause always comes after the FROM statement!

Note that in this course we will use <> and not != for the not equal operator, as per the SQL standard.

What does the following query return?

```
SELECT title
FROM films
WHERE release_year > 2000;
```

**Possible Answers**

- [ ] Films released before the year 2000
- [x] Films released after the year 2000
- [ ] Films released after the year 2001
- [ ] Films released in 2000

**Simple filtering of numeric values**

```
SELECT *
FROM films
WHERE release_year = 2016;

SELECT COUNT(*)
FROM films
WHERE release_year < 2000;

SELECT title, release_year
FROM films
WHERE release_year > 2000;
```

**Simple filtering of text**

```
SELECT *
FROM films
WHERE language = 'French';

SELECT name, birthdate
FROM people
WHERE birthdate = '1974-11-11';

SELECT COUNT(*)
FROM films
WHERE language = 'Hindi';

SELECT *
FROM films
WHERE certification = 'R';
```

**WHERE AND**

```
SELECT title, release_year
FROM films
WHERE release_year < 2000
AND language = 'Spanish';

SELECT *
FROM films
WHERE release_year > 2000
AND language = 'Spanish';

SELECT *
FROM films
WHERE release_year > 2000
AND release_year < 2010
AND language = 'Spanish';
```

**WHERE AND OR**

What if you want to select rows based on multiple conditions where some but not all of the conditions need to be met? 

For this, SQL has the OR operator.

For example, the following returns all films released in either 1994 or 2000:

```
SELECT title
FROM films
WHERE release_year = 1994
OR release_year = 2000;
```

Note that you need to specify the column for every OR condition, so the following is invalid:

```
SELECT title
FROM films
WHERE release_year = 1994 OR 2000;
```

When combining AND and OR, be sure to enclose the individual clauses in parentheses, like so:

```
SELECT title
FROM films
WHERE (release_year = 1994 OR release_year = 1995)
AND (certification = 'PG' OR certification = 'R');
```

Otherwise, due to SQL's precedence rules, you may not get the results you're expecting!

What does the OR operator do?

**Possible Answers**

- [x] Display only rows that meet at least one of the specified conditions.
- [ ] Display only rows that meet all of the specified conditions.
- [ ] Display only rows that meet none of the specified conditions.

**WHERE AND OR (2)**

```
SELECT title, release_year
FROM films
WHERE release_year >= 1990 AND release_year < 2000;

SELECT title, release_year
FROM films
WHERE (release_year >= 1990 AND release_year < 2000)
AND (language = 'French' OR language = 'Spanish');

SELECT title, release_year
FROM films
WHERE (release_year >= 1990 AND release_year < 2000)
AND (language = 'French' OR language = 'Spanish')
AND gross > 2000000;
```

**BETWEEN**

As you've learned, you can use the following query to get titles of all films released in and between 1994 and 2000:

```
SELECT title
FROM films
WHERE release_year >= 1994
AND release_year <= 2000;
```

Checking for ranges like this is very common, so in SQL the BETWEEN keyword provides a useful shorthand for filtering values within a specified range. 

This query is equivalent to the one above:

```
SELECT title
FROM films
WHERE release_year
BETWEEN 1994 AND 2000;
```

It's important to remember that BETWEEN is inclusive, meaning the beginning and end values are included in the results!

What does the BETWEEN keyword do?

**Possible Answers**

- [ ] Filter numeric values
- [ ] Filter text values
- [ ] Filter values in a specified list
- [x] Filter values in a specified range

**BETWEEN (2)**

```
SELECT title, release_year
FROM films
WHERE release_year BETWEEN 1990 AND 2000;

SELECT title, release_year
FROM films
WHERE release_year BETWEEN 1990 AND 2000
AND budget > 100000000;

SELECT title, release_year
FROM films
WHERE release_year BETWEEN 1990 AND 2000
AND budget > 100000000
AND language = 'Spanish';

SELECT title, release_year
FROM films
WHERE release_year BETWEEN 1990 AND 2000
AND budget > 100000000
AND (language = 'Spanish' OR language = 'French');
```

**WHERE IN**

```
SELECT title, release_year
FROM films
WHERE release_year IN (1990, 2000)
AND duration > 120;

SELECT title, language
FROM films
WHERE language IN ('English', 'Spanish', 'French');

SELECT title, certification
FROM films
WHERE certification IN ('NC-17', 'R');
```

**Introduction to NULL and IS NULL**

In SQL, NULL represents a missing or unknown value. You can check for NULL values using the expression IS NULL. 

For example, to count the number of missing birth dates in the people table:

```
SELECT COUNT(*)
FROM people
WHERE birthdate IS NULL;
```

As you can see, IS NULL is useful when combined with WHERE to figure out what data you're missing.

Sometimes, you'll want to filter out missing values so you only get results which are not NULL. To do this, you can use the IS NOT NULL operator.

For example, this query gives the names of all people whose birth dates are not missing in the people table.

```
SELECT name
FROM people
WHERE birthdate IS NOT NULL;
```

What does NULL represent?

**Possible Answers**

- [ ] A corrupt entry
- [x] A missing value
- [ ] An empty string
- [ ] An invalid value

**NULL and IS NULL**

```
SELECT name
FROM people
WHERE deathdate IS NULL;

SELECT title
FROM films
WHERE budget IS NULL;

SELECT COUNT(*)
FROM films
WHERE language IS NULL;
```

**LIKE and NOT LIKE**

```
SELECT name
FROM people
WHERE name LIKE 'B%';

SELECT name
FROM people
WHERE name LIKE '_r%';

SELECT name
FROM people
WHERE name NOT LIKE 'A%';
```

## 3. Aggregate Functions

**Aggregate functions**

```
SELECT SUM(duration)
FROM films;

SELECT AVG(duration)
FROM films;

SELECT MIN(duration)
FROM films;

SELECT MAX(duration)
FROM films;
```

**Aggregate functions practice**

```
SELECT SUM(gross)
FROM films;

SELECT AVG(gross)
FROM films;

SELECT MIN(gross)
FROM films;

SELECT MAX(gross)
FROM films;
```

**Combining aggregate functions with WHERE**

```
SELECT SUM(gross)
FROM films
WHERE release_year >= 2000;

SELECT AVG(gross)
FROM films
where title LIKE 'A%';

SELECT MIN(gross)
FROM films
WHERE release_year = 1994;

SELECT MAX(gross)
FROM films
WHERE release_year BETWEEN 2000 AND 2012;
```

**A note on arithmetic**

In addition to using aggregate functions, you can perform basic arithmetic with symbols like +, -, *, and /.

So, for example, this gives a result of 12:

`SELECT (4 * 3);`

However, the following gives a result of 1:

`SELECT (4 / 3);`

What's going on here?

SQL assumes that if you divide an integer by an integer, you want to get an integer back. So be careful when dividing!

If you want more precision when dividing, you can add decimal places to your numbers. For example,

`SELECT (4.0 / 3.0) AS result;`

gives you the result you would expect: 1.333.

What is the result of `SELECT (10 / 3);`?

**Possible Answers**

- [ ] 2.333
- [ ] 3.333
- [x] 3
- [ ] 3.0

**It's AS simple AS aliasing**

```
SELECT title, gross - budget AS net_profit
FROM films;

SELECT title, duration / 60.0 AS duration_hours
FROM films;

SELECT AVG(duration) / 60.0 AS avg_duration_hours  
FROM films;
```

**Even more aliasing**

```
-- get the count(deathdate) and multiply by 100.0
-- then divide by count(*) 
SELECT COUNT(deathdate) * 100.0 / COUNT(*) AS percentage_dead
FROM people;

SELECT MAX(release_year) - MIN(release_year)
AS difference
FROM films;

SELECT (MAX(release_year) - MIN(release_year)) / 10.0
AS number_of_decades
FROM films;
```

## 4. Sorting and grouping

**ORDER BY**

In SQL, the ORDER BY keyword is used to sort results in ascending or descending order according to the values of one or more columns.

By default ORDER BY will sort in ascending order. If you want to sort the results in descending order, you can use the DESC keyword. 

For example,

```
SELECT title
FROM films
ORDER BY release_year DESC;
```

gives you the titles of films sorted by release year, from newest to oldest.

How do you think ORDER BY sorts a column of text values by default?

**Possible Answers**

- [x] Alphabetically (A-Z)
- [ ] Reverse alphabetically (Z-A)
- [ ] There's no natural ordering to text data
- [ ] By number of characters (fewest to most)

**Sorting single columns**

```
SELECT name
FROM people
ORDER BY name;

SELECT name
FROM people
ORDER BY birthdate;

SELECT birthdate, name
FROM people
ORDER BY birthdate;
```

**Sorting single columns (2)**

```
SELECT title
FROM films
WHERE release_year IN (2000, 2012)
ORDER BY release_year;

SELECT *
FROM films
WHERE release_year <> 2015
ORDER BY duration;

SELECT title, gross
FROM films
WHERE title LIKE 'M%'
ORDER BY title;
```

**Sorting single columns (DESC)**

```
SELECT imdb_score, film_id
FROM reviews
ORDER BY imdb_score DESC;

SELECT title
FROM films
ORDER BY title DESC;

SELECT title, duration
FROM films
ORDER BY duration DESC;
```

**Sorting multiple columns**

```
SELECT birthdate, name
FROM people
ORDER BY birthdate, name;

SELECT release_year, duration, title
FROM films
ORDER BY release_year, duration;

SELECT certification, release_year, title
FROM films
ORDER BY certification, release_year;

SELECT name, birthdate
FROM people
ORDER BY name, birthdate;
```

**GROUP BY**

Now you know how to sort results! Often you'll need to aggregate results. For example, you might want to count the number of male and female employees in your company. Here, what you want is to group all the males together and count them, and group all the females together and count them. 

In SQL, GROUP BY allows you to group a result by one or more columns, like so:

```
SELECT sex, count(*)
FROM employees
GROUP BY sex;
```

This might give, for example:

```
sex	count
male	15
female	19
```

Commonly, GROUP BY is used with aggregate functions like COUNT() or MAX(). Note that GROUP BY always goes after the FROM clause!

What is GROUP BY used for?

**Possible Answers**

- [ ] Performing operations by column
- [ ] Performing operations all at once
- [ ] Performing operations in a particular order
- [x] Performing operations by group

**GROUP BY practice**

```
SELECT release_year, COUNT(*)
FROM films
GROUP BY release_year;

SELECT release_year, AVG(duration)
FROM films
GROUP BY release_year;

SELECT release_year, MAX(budget)
FROM films
GROUP BY release_year;

SELECT imdb_score, COUNT(*)
FROM reviews
GROUP BY imdb_score;
```

**GROUP BY practice (2)**

```
SELECT release_year, MIN(gross)
FROM films
GROUP BY release_year;

SELECT language, SUM(gross)
FROM films
GROUP BY language;

SELECT country, SUM(budget)
FROM films
GROUP BY country;

SELECT release_year, country, MAX(budget)
FROM films
GROUP BY release_year, country
ORDER BY release_year, country;

SELECT country, release_year, MIN(gross)
FROM films
GROUP BY country, release_year
ORDER BY country, release_year;
```

**HAVING a great time**

In SQL, aggregate functions can't be used in WHERE clauses. 

For example, the following query is invalid:

```
SELECT release_year
FROM films
GROUP BY release_year
WHERE COUNT(title) > 10;
```

This means that if you want to filter based on the result of an aggregate function, you need another way! That's where the HAVING clause comes in. For example,

```
SELECT release_year
FROM films
GROUP BY release_year
HAVING COUNT(title) > 10;
```

shows only those years in which more than 10 films were released.

In how many different years were more than 200 movies released?

**Possible Answers**

- [ ] 2
- [x] 13
- [ ] 44
- [ ] 63

**All together now**

```
SELECT release_year, budget, gross
FROM films;

SELECT release_year, budget, gross
FROM films
WHERE release_year > 1990;

SELECT release_year
FROM films
WHERE release_year > 1990
GROUP BY release_year;

SELECT release_year, AVG(budget) AS avg_budget, AVG(gross) AS avg_gross
FROM films
WHERE release_year > 1990
GROUP BY release_year;

SELECT release_year, AVG(budget) AS avg_budget, AVG(gross) AS avg_gross
FROM films
WHERE release_year > 1990
GROUP BY release_year
HAVING AVG(budget) > 60000000;

SELECT release_year, AVG(budget) AS avg_budget, AVG(gross) AS avg_gross
FROM films
WHERE release_year > 1990
GROUP BY release_year
HAVING AVG(budget) > 60000000
ORDER BY avg_gross DESC;
```

**All together now (2)**

```
-- select country, average budget, 
-- and average gross
SELECT country, AVG(budget) AS avg_budget, 
       AVG(gross) AS avg_gross
-- from the films table
FROM films
-- group by country 
GROUP BY country
-- where the country has more than 10 titles
HAVING COUNT(title) > 10
-- order by country
ORDER BY country
-- limit to only show 5 results
LIMIT 5;
```

**A taste of things to come**

Congrats on making it to the end of the course! By now you should have a good understanding of the basics of SQL.

There's one more concept we're going to introduce. You may have noticed that all your results so far have been from just one table, e.g. films or people.

In the real world however, you will often want to query multiple tables. For example, what if you want to see the IMDB score for a particular movie?

In this case, you'd want to get the ID of the movie from the films table and then use it to get IMDB information from the reviews table. In SQL, this concept is known as a join, and a basic join is shown in the editor to the right.

The query in the editor gets the IMDB score for the film To Kill a Mockingbird! Cool right?

As you can see, joins are incredibly useful and important to understand for anyone using SQL.

We have a whole follow-up course dedicated to them called Joining Data in PostgreSQL for you to hone your database skills further!

**Question**

What is the IMDB score for the film To Kill a Mockingbird?

```
SELECT title, imdb_score
FROM films
JOIN reviews
ON films.id = reviews.film_id
WHERE title = 'To Kill a Mockingbird';
```

**Possible Answers**

- [ ] 8.1
- [x] 8.4
- [ ] 7.7
- [ ] 9.3
"""