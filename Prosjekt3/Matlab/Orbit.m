startTime = datetime(2020,5,5,0,0,0);
stopTime = startTime + days(1);
sampleTime = 60;        
sc = satelliteScenario(startTime,stopTime,sampleTime);

tleFile = "ISS_TLE.txt";
ISS = satellite(sc,tleFile,"Name","ISS");

semiMajorAxis = (6378.137 + 400)*10^3;                                                      %m
eccentricity = 0;
inclination = 97;                                                                           %degrees
rightAscensionOfAscendingNode = -45;                                                        %degrees
argumentOfPeriapsis = 0;                                                                    %degrees
trueAnomaly = 0;                                                                            %degrees
sat = satellite(sc,semiMajorAxis,eccentricity,inclination,rightAscensionOfAscendingNode,...
    argumentOfPeriapsis,trueAnomaly,"OrbitPropagator","two-body-keplerian","Name","Sat");


ac = access(ISS,sat);
accessIntervals(ac)

play(sc)