drop database if exists Micro;
create database Micro;
use Micro;

create table users (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    email VARCHAR(50), 
    password VARCHAR(50)
    );
