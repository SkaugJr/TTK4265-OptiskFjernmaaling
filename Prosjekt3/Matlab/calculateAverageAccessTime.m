function calculateAverageAccessTime(downlinkAccess, startTime, stopTime)
    % Beregn antall dager basert på start- og sluttid
    simulationDays = days(stopTime - startTime);
    
    intervals = accessIntervals(downlinkAccess);
    
    % Hent ut start- og sluttidene som datetime-variabler
    startTimes = intervals.StartTime;
    endTimes = intervals.EndTime;
    
    % Beregn varighet for hver tilgang
    durations = endTimes - startTimes;
    
    % Gjennomsnittlig downlink tid per periode
    avgTimePerPeriod = mean(durations);
    
    % Total nedlink-tid i hele simuleringen
    totalTimeInSimulation = sum(durations);
    
    % Gjennomsnittlig nedlink-tid per dag
    avgTimePerDay = totalTimeInSimulation / simulationDays;
    
    % Skriv ut resultatene
    fprintf('Gjennomsnittlig kontakttid per periode: %s\n', avgTimePerPeriod);
    fprintf('Gjennomsnittlig kontakttid per dag: %s\n', avgTimePerDay);
    fprintf('Total kontakttid i hele simuleringen: %s\n', totalTimeInSimulation);
end
