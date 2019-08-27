var CeleryProgressBar = (function () {

    function onSuccessDefault(progressBarElement, progressBarMessageElement) {
        progressBarElement.style.backgroundColor = '#76ce60';
        progressBarMessageElement.innerHTML = "Success!";
    }

    function onErrorDefault(progressBarElement, progressBarMessageElement) {
        progressBarElement.style.backgroundColor = '#dc4f63';
        progressBarMessageElement.innerHTML = "Uh-Oh, something went wrong!";
    }

    function onProgressDefault(progressBarElement, progressBarMessageElement, progressBarPeakElement, progress) {
        progressBarElement.style.backgroundColor = '#68a9ef';
        progressBarElement.style.width = progress.percent + "%";
        progressBarMessageElement.innerHTML = progress.current + ' of ' + progress.total + ' processed.';
        if (parseInt(progress.total) > parseInt(progressBarPeakElement.innerHTML)) {
            progressBarPeakElement.innerHTML = progress.total;
        }
    }

    function updateProgress (progressUrl, options) {
        options = options || {};
        let progressBarId = options.progressBarId || 'progress-bar';
        let progressBarMessage = options.progressBarMessageId || 'progress-bar-message';
        let progressBarPeakTotal = options.progressBarPeakTotal || 'progress-bar-peak-total';
        let progressBarElement = options.progressBarElement || document.getElementById(progressBarId);
        let progressBarMessageElement = options.progressBarMessageElement || document.getElementById(progressBarMessage);
        let progressBarPeakElement = options.progressBarPeakElement || document.getElementById(progressBarPeakTotal);
        let onProgress = options.onProgress || onProgressDefault;
        let onSuccess = options.onSuccess || onSuccessDefault;
        let onError = options.onError || onErrorDefault;
        let pollInterval = options.pollInterval || 500;

        fetch(progressUrl).then(function(response) {
            response.json().then(function(data) {
                if (data.progress) {
                    onProgress(progressBarElement, progressBarMessageElement, progressBarPeakElement, data.progress);
                }
                if (!data.complete) {
                    setTimeout(updateProgress, pollInterval, progressUrl, options);
                } else {
                    if (data.success) {
                        onSuccess(progressBarElement, progressBarMessageElement);
                    } else {
                        onError(progressBarElement, progressBarMessageElement);
                    }
                }
            });
        });
    }

    return {
        onSuccessDefault: onSuccessDefault,
        onErrorDefault: onErrorDefault,
        onProgressDefault: onProgressDefault,
        updateProgress: updateProgress,
        initProgressBar: updateProgress,  // just for api cleanliness
    };
})();

var CelerySpinner = (function () {

    function onSuccessDefault(spinnerElement, dataForLater) {
        spinnerElement.style.color = '#76ce60';
        spinnerElement.innerHTML = '<i class="fa fa-smile"></i>';
    }

    function onErrorDefault(spinnerElement) {
        spinnerElement.style.color = '#dc4f63';
        spinnerElement.innerHTML = '<i class="fa fa-poo-storm"></i>';
    }

    function onProgressDefault(spinnerInnerElement, progress) {
        spinnerInnerElement.innerHTML = Math.round(progress.percent) + '%';
    }

    function updateProgress(progressUrl, options) {
        options = options || {};
        let spinnerId = options.spinnerId || 'spinner';
        let spinnerInnerId = spinnerId + "_inner";
        let spinnerElement = options.spinnerElement || document.getElementById(spinnerId);
        let spinnerInnerElement = options.spinnerInnerElement || document.getElementById(spinnerInnerId);
        let onProgress = options.onProgress || onProgressDefault;
        let onSuccess = options.onSuccess || onSuccessDefault;
        let onError = options.onError || onErrorDefault;
        let pollInterval = options.pollInterval || 500;
        let alreadyStarted = options.alreadyStarted || "False";
        let dataForLater = options.dataForLater || "";

        if(alreadyStarted === "False") {
            // Setup that only needs to happen once: (repeated code lives in onProgress/onProgressDefault
            spinnerElement.style.color = '#68a9ef';
            html_for_spinner = '<div style="text-align: center;"><span class="fa-stack fa-4x">';
            html_for_spinner += '<i class="fa fa-spinner fa-spin fa-stack-2x"></i>';
            html_for_spinner += '<strong id="' + spinnerInnerId + '" class="fa-stack-1x pct_text"></strong>';
            html_for_spinner += '</span></div>';
            spinnerElement.innerHTML = html_for_spinner;
            spinnerInnerElement = document.getElementById(spinnerInnerId);
            options.alreadyStarted = "True";
        }

        fetch(progressUrl).then(function(response) {
            response.json().then(function(data) {
                if (data.progress) {
                    onProgress(spinnerInnerElement, data.progress);
                }
                if (!data.complete) {
                    setTimeout(updateProgress, pollInterval, progressUrl, options);
                } else {
                    if (data.success) {
                        onSuccess(spinnerElement, dataForLater);
                    } else {
                        onError(spinnerElement);
                    }
                }
            });
        });
    }

    return {
        onSuccessDefault: onSuccessDefault,
        onErrorDefault: onErrorDefault,
        onProgressDefault: onProgressDefault,
        updateProgress: updateProgress,
        initSpinner: updateProgress
    };
})();
