-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Feb 24, 2024 at 01:32 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `3vehicleparknumdb`
--

-- --------------------------------------------------------

--
-- Table structure for table `amttb`
--

CREATE TABLE `amttb` (
  `id` bigint(10) NOT NULL auto_increment,
  `UserName` varchar(250) NOT NULL,
  `VehicleNo` varchar(250) NOT NULL,
  `Date` varchar(250) NOT NULL,
  `Amount` double(10,2) NOT NULL,
  `Status` varchar(250) NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

--
-- Dumping data for table `amttb`
--


-- --------------------------------------------------------

--
-- Table structure for table `entrytb`
--

CREATE TABLE `entrytb` (
  `id` bigint(20) NOT NULL auto_increment,
  `VehicleNo` varchar(250) NOT NULL,
  `Date` varchar(250) NOT NULL,
  `InTime` time NOT NULL,
  `OutTime` time NOT NULL,
  `Status` varchar(250) NOT NULL,
  `ParkingNo` varchar(250) NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

--
-- Dumping data for table `entrytb`
--


-- --------------------------------------------------------

--
-- Table structure for table `regtb`
--

CREATE TABLE `regtb` (
  `id` bigint(10) NOT NULL auto_increment,
  `Name` varchar(250) NOT NULL,
  `Mobile` varchar(250) NOT NULL,
  `Email` varchar(250) NOT NULL,
  `VehicleNo` varchar(50) NOT NULL,
  `UserName` varchar(250) NOT NULL,
  `Password` varchar(250) NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

--
-- Dumping data for table `regtb`
--


-- --------------------------------------------------------

--
-- Table structure for table `temptb`
--

CREATE TABLE `temptb` (
  `id` int(10) NOT NULL auto_increment,
  `uname` varchar(250) NOT NULL,
  `VehicleNo` varchar(250) NOT NULL,
  `Mobile` varchar(250) NOT NULL,
  `Email` varchar(250) NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

--
-- Dumping data for table `temptb`
--

