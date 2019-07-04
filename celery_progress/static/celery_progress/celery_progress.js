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
