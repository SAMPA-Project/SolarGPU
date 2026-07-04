#pragma once

/* Hourly sun azimuth/altitude for a full (non-leap) year at a fixed
   lon/lat/timezone, via the PSA solar position algorithm (Blanco-Muriel et
   al., "Computing the solar vector", Solar Energy 80(3), 2006). Accuracy is
   ~0.01 deg, which is plenty for hourly irradiance over a DSM. */

#include <cmath>
#include <ctime>
#include <tuple>
#include <vector>

namespace solar_gpu {

using SunSample = std::tuple<float, float>;

class SunPositionCalculator {
public:
    int timezone = 0;

    SunPositionCalculator(float lon, float lat, int tz=0)
        : timezone(tz), lon_(lon), lat_(lat) {}

    void calc(std::vector<SunSample>& out) {
        const int totalDays = is_leap_year(year_) ? 366 : 365;
        out.reserve((size_t)totalDays * 24);

        for (int dayOfYear = 1; dayOfYear <= totalDays; ++dayOfYear) {
            std::tm date{};
            date.tm_year = year_ - 1900;
            date.tm_mday = dayOfYear;
            std::mktime(&date);

            for (int hour = 0; hour < 24; ++hour) {
                float az = 0.0f, alt = 0.0f;
                psa_calc(date.tm_mon + 1, date.tm_mday, hour, /*minute=*/0, az, alt);
                out.emplace_back(az, alt);
            }
        }
    }

    static bool is_leap_year(unsigned year) {
        return (year % 400 == 0) || (year % 4 == 0 && year % 100 != 0);
    }

private:
    float lon_ = 0.0f, lat_ = 0.0f;
    int year_ = 2026;

    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double kRad = kPi / 180.0;
    static constexpr double kTwoPi = 2.0 * kPi;
    static constexpr double kEarthMeanRadiusKm = 6371.01;
    static constexpr double kAstronomicalUnitKm = 149597890.0;

    /* PSA algorithm; az/alt in degrees. hour/minute are local time, converted
       to UT via the fixed `timezone` offset. */
    void psa_calc(int month, int day, int hour, int minute, float& az, float& alt) {
        double utcHours = hour - timezone + minute / 60.0;

        // Elapsed Julian days since 2000-01-01 12:00 UT.
        long liAux1 = (month - 14) / 12;
        long liAux2 = (1461 * (year_ + 4800 + liAux1)) / 4
            + (367 * (month - 2 - 12 * liAux1)) / 12
            - (3 * ((year_ + 4900 + liAux1) / 100)) / 4
            + day - 32075;
        double elapsedJulianDays = (double)liAux2 - 0.5 + utcHours / 24.0 - 2451545.0;

        // Ecliptic coordinates.
        double omega = 2.1429 - 0.0010394594 * elapsedJulianDays;
        double meanLongitude = 4.8950630 + 0.017202791698 * elapsedJulianDays;
        double anomaly = 6.2400600 + 0.0172019699 * elapsedJulianDays;
        double eclipticLongitude = meanLongitude + 0.03341607 * std::sin(anomaly)
            + 0.00034894 * std::sin(2 * anomaly) - 0.0001134 - 0.0000203 * std::sin(omega);
        double eclipticObliquity = 0.4090928 - 6.2140e-9 * elapsedJulianDays + 0.0000396 * std::cos(omega);

        // Celestial coordinates (right ascension, declination).
        double sinEclipticLongitude = std::sin(eclipticLongitude);
        double dY = std::cos(eclipticObliquity) * sinEclipticLongitude;
        double dX = std::cos(eclipticLongitude);
        double rightAscension = std::atan2(dY, dX);
        if (rightAscension < 0.0) rightAscension += kTwoPi;
        double declination = std::asin(std::sin(eclipticObliquity) * sinEclipticLongitude);

        // Local coordinates (azimuth, zenith).
        double greenwichMeanSiderealTime = 6.6974243242 + 0.0657098283 * elapsedJulianDays + utcHours;
        double localMeanSiderealTime = (greenwichMeanSiderealTime * 15.0 + lon_) * kRad;
        double hourAngle = localMeanSiderealTime - rightAscension;
        double latitudeRad = lat_ * kRad;
        double cosLatitude = std::cos(latitudeRad);
        double sinLatitude = std::sin(latitudeRad);
        double cosHourAngle = std::cos(hourAngle);

        double zenith = std::acos(cosLatitude * cosHourAngle * std::cos(declination) + std::sin(declination) * sinLatitude);
        double azY = -std::sin(hourAngle);
        double azX = std::tan(declination) * cosLatitude - sinLatitude * cosHourAngle;
        double azimuth = std::atan2(azY, azX);
        if (azimuth < 0.0) azimuth += kTwoPi;
        azimuth /= kRad;

        // Parallax correction (topocentric vs geocentric zenith).
        double parallax = (kEarthMeanRadiusKm / kAstronomicalUnitKm) * std::sin(zenith);
        zenith = (zenith + parallax) / kRad;

        alt = (float)(90.0 - zenith);
        az = (float)azimuth;
    }
};

} // namespace solar_gpu
