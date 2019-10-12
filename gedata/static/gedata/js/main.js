// Local support functions

function caption(figure) {
    if( figure === 2) {
        return '<p><span class="heavy">Maximizing Mantel correlations between gene expression similarity ' +
            'and functional connectivity similarity by greedily removing genes.</span><br />' +
            '<span class="heavy">A)</span> Mantel correlations for gene expression similarity matrices created ' +
            'from all 15,745 genes are all roughly around zero, as shown in the left-most pane. ' +
            'Repeatedly dropping the gene least supportive ' +
            'of a positive correlation drives the Mantel correlation higher, to a peak, after which dropping ' +
            'any gene results in lower correlation. Black/gray data represent training data, randomly split in half' +
            'by sample. The split-half training data then had sample labels shuffled randomly (green, right-most), ' +
            'weighted to preserve distance (red, center-right), or had its edges shuffled within distance bins ' +
            '(magenta, center-left). Each shuffling paradigm was applied 16 times, each with a different seed. ' +
            'Shuffled data were then subjected to the same Mantel maximization algorithm. Peak Mantel correlations ' +
            'for each set are shown in the right-most pane.<br />' +
            '<span class="heavy">B)</span> Genes remaining at the peak of each training were more consistent in ' +
            'real data than in shuffled data. Training on randomly shuffled data resulted in randomly selected ' +
            'genes, with low similarity.<br />' +
            '<span class="heavy">C)</span> Filtering actual data, without shuffling, by the genes discovered in ' +
            'the training phase (with real, shuffled, and/or masked data), resulted in slightly lower correlations, ' +
            'but genes discovered in real data drove higher Mantel correlations than genes discovered in shuffled ' +
            'data. This was true in training data (left-most pane), training data with edges nearer than 16mm ' +
            '(center-left pane), an independent test set (center-right pane), and the test set with proximal edges ' +
            '(< 16mm) removed (right-most pane).' +
            '</p>';
    } else if( figure === 3 ) {
        return '<p><span class="heavy">Assess algorithm performance at different thresholds.</span><br />' +
            '<span class="heavy">Peak)</span> The highest Mantel correlation achieved during training, with the ' +
            'specified training data, masked, shuffled, or otherwise manipulated.<br />' +
            '<span class="heavy">Train or Test)</span> The Mantel correlation of gene expression similarity, ' +
            'filtered to include only the top probes at a given threshold, and connectivity similarity. These curves ' +
            'are in real data, regardless of the training manipulations. At the threshold responsible for the peak ' +
            'Mantel correlation, in unshuffled and unmasked data, the train line should meet the peak line. See ' +
            '<a href="https://github.com/mfschmidt/ge_data_manager/blob/master/gedata/tasks.py" target="_blank">' +
            '<code>tasks.py:test_score</code></a>.<br />' +
            '<span class="heavy">Overlap)</span> At each threshold, the top probes discovered in the training half ' +
            'and the top probes discovered in the test half have some probes in common. This overlap is the percent ' +
            'similarity across split-halves. In other words, This percentage of probes survived beyond the threshold ' +
            'in both split-halves. See ' +
            '<a href="https://github.com/mfschmidt/ge_data_manager/blob/master/gedata/tasks.py" target="_blank">' +
            '<code>tasks.py:test_overlap</code></a>.' +
            '</p>';
    } else {
        return '<p></p>';
    }
}

function startResultMining(job_name) {
    document.getElementById("celery_progress_bar").style.display = "block";
    document.getElementById("static_progress_bar").style.display = "none";
    let refreshUrl = "/gedata/REST/refresh/" + job_name;
    let request = new XMLHttpRequest();
    request.onreadystatechange = function () {
        if (request.readyState === 4 && request.status === 200) {
            let responseJsonObj = JSON.parse(request.responseText);
            console.log("Got id: " + responseJsonObj.task_id);
            let progressUrl = "/celery-progress/" + responseJsonObj.task_id + "/";
            CeleryProgressBar.initProgressBar(progressUrl, {
                onSuccess: stopCeleryProcessPolling
            });
        }
    };
    request.open("GET", refreshUrl, true);
    request.send();
    console.log("Beginning celery polling for " + job_name + ".");
}

function stopCeleryProcessPolling() {
    document.getElementById("progress-bar").style.backgroundColor = '#76ce60';
    let peak = document.getElementById("progress-bar-peak-total").innerHTML.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    document.getElementById("progress-bar-message").innerHTML = "Found " + peak + " results.";
    document.getElementById("celery_progress_bar").style.display = "none";
    document.getElementById("static_progress_bar").style.display = "block";
    location.reload();
}

function shouldBailOnBuilding(image_element, select_element) {
    // If a pane is busy, ignore new build requests.

    if( image_element.innerHTML.includes(select_element.innerText)) {
        // The desired image is already built and loaded. Nothing to do here.
        console.log("  doing nothing with " + select_element.innerText + "; it's already built and loaded.");
        return true;
    } else if( image_element.innerHTML.includes("fa-spinner")) {
        // Already working on it; ignore further requests.
        // But with async, many requests may be made for the same thing before this is triggered.
        console.log("  " + select_element.id + " is already building a plot. Wait until it's done!");
        return true;
    } else if (image_element.innerHTML.includes("img src")) {
        // Immediately set the image blank, just to give feedback we registered the click.
        // But don't mess with active spinners (which do not contain "img src" substring)
        console.log("  priming the " + select_element.id.toUpperCase()[0] + " image spot for " + select_element.innerText + " with empty.");
        loadPlot(image_element, "/static/gedata/empty.png");
    }
    return false;
}

function buildPlot(image_id, select_id) {
    // Calculate the image id string from forms, then use it to load plot images
    let select_element = document.getElementById(select_id);
    select_element.innerText = image_id_from_selections(select_id.toUpperCase()[0]);  // "L" or "R"
    console.log("in buildPlot, select_id = " + select_id + "; containing '" + select_element.innerText + "'.");
    let image_element = document.getElementById(image_id);

    if(shouldBailOnBuilding(image_element, select_element)) {
        return;
    }

    console.log("Checking for " + select_element.innerText + " image for " + image_id + ".");
    let img_file = "train_test_" + select_element.innerText.toLowerCase() + ".png";
    let img_url = "/static/gedata/plots/" + img_file;

    // The first ajax request determines whether our desired plot already exists or not.
    let png_http = new XMLHttpRequest();
    png_http.onreadystatechange = function () {
        if (png_http.readyState === 4) {
            if (png_http.status === 200) {
                console.log(select_element.innerText + " exists, loading it rather than rebuilding.");
                loadPlot(image_element, img_url);
            }
            if (png_http.status === 404) {
                console.log(select_element.innerText + " does not exist. Building it from scratch.");
                // let refreshUrl = "{% url 'gedata:refresh' job_name=12345 %}".replace(/12345/, img_file);
                let refreshUrl = "/gedata/REST/refresh/" + img_file;

                // The second ajax request initiates image creation and starts the spinner.
                let request = new XMLHttpRequest();
                request.onreadystatechange = function () {
                    if (request.readyState === 4 && request.status === 200) {
                        let responseJsonObj = JSON.parse(this.responseText);
                        if (responseJsonObj.task_id !== "None") {
                            console.log("Got id: " + responseJsonObj.task_id);
                            let progressUrl = "/celery-progress/" + responseJsonObj.task_id + "/";
                            CelerySpinner.initSpinner(progressUrl, {
                                onSuccess: loadPlot,
                                spinnerId: image_id,
                                dataForLater: img_url
                            });
                        }
                    }
                };
                request.open("GET", refreshUrl, true);
                request.send();
                console.log("Beginning to build plot for " + select_element.innerText + " at " + image_id);
            }
        }
    };
    png_http.open('HEAD', img_url, true);
    png_http.send();
}

function assessPerformance(image_id, select_id) {
    // Calculate the image id string from forms, then use it to load plot images
    let select_element = document.getElementById(select_id);
    select_element.innerText = image_id_from_selections(select_id.toLowerCase());  // "performance"
    console.log("in assessPerformance, select_id = " + select_id + "; containing '" + select_element.innerText + "'.");
    let image_element = document.getElementById(image_id);

    if(shouldBailOnBuilding(image_element, select_element)) {
        return;
    }

    console.log("Checking for " + select_element.innerText + " image for " + image_id + ".");
    let img_file = "performance_" + select_element.innerText.toLowerCase() + ".png";
    let img_url = "/static/gedata/plots/" + img_file;

    // The first ajax request determines whether our desired plot already exists or not.
    let png_http = new XMLHttpRequest();
    png_http.onreadystatechange = function () {
        if (png_http.readyState === 4) {
            if (png_http.status === 200) {
                console.log(select_element.innerText + " exists, loading it rather than rebuilding.");
                loadPlot(image_element, img_url);
            }
            if (png_http.status === 404) {
                console.log(select_element.innerText + " does not exist. Building it from scratch.");
                // let refreshUrl = "{% url 'gedata:refresh' job_name=12345 %}".replace(/12345/, img_file);
                let refreshUrl = "/gedata/REST/refresh/" + img_file;

                // The second ajax request initiates image creation and starts the spinner.
                let request = new XMLHttpRequest();
                request.onreadystatechange = function () {
                    if (request.readyState === 4 && request.status === 200) {
                        let responseJsonObj = JSON.parse(this.responseText);
                        if (responseJsonObj.task_id !== "None") {
                            console.log("Got id: " + responseJsonObj.task_id);
                            let progressUrl = "/celery-progress/" + responseJsonObj.task_id + "/";
                            CelerySpinner.initSpinner(progressUrl, {
                                onSuccess: loadPlot,
                                spinnerId: image_id,
                                dataForLater: img_url
                            });
                        }
                    }
                };
                request.open("GET", refreshUrl, true);
                request.send();
                console.log("Beginning to assess performance for " + select_element.innerText + " at " + image_id);
            }
        }
    };
    png_http.open('HEAD', img_url, true);
    png_http.send();
}

function loadPlot(image_element, image_url) {
    let w = Math.round(document.documentElement.clientWidth * 0.45);
    image_element.style.color = '#ffffff';
    let html_text = "<a href=\"" + image_url + "\">";
    html_text += "<img src=\"" + image_url + "\" width=\"" + w + "\" alt=\"" + image_url + "\">";
    html_text += "</a>";
    image_element.innerHTML = html_text;

    // Also update gene ranking information, if available.
    if( image_url.includes("train_test_") ) {
        if (image_url.endsWith('empty.png')) {
            document.getElementById(image_element.id.replace('image', 'go')).innerHTML = "";
        } else {
            append_probes_from_file(image_element.id.replace('image', 'go'), image_url.replace('png', 'html'));
            document.getElementById(image_element.id.replace('image', 'caption')).innerHTML = caption(2);
        }
    }
    if( image_url.includes("performance_") ) {
        if (!image_url.endsWith('empty.png')) {
            document.getElementById(image_element.id.replace('image', 'caption')).innerHTML = caption(3);
        }
    }
}

function load_file(method, url) {
    return new Promise(function (resolve, reject) {
        let xhr = new XMLHttpRequest();
        xhr.open(method, url);
        xhr.onload = function () {
            if (this.status >= 200 && this.status < 300) {
                resolve(xhr.response);
            } else {
                reject({
                    status: this.status,
                    statusText: xhr.statusText
                });
            }
        };
        xhr.onerror = function () {
            reject({
                status: this.status,
                statusText: xhr.statusText
            });
        };
        xhr.send();
    });
}

async function append_probes_from_file(elementID, text_url) {
    console.log(text_url);
    let result = await load_file("GET", text_url);
    document.getElementById(elementID).innerHTML += result;
}

function image_id_from_selections(side) {
    // For "L" and "R" side for either the "Compare result sets" or "Result set comparisons" page, return the url
    // to the appropriate image, based on form selections.
    let summary_string = "";
    if (side === "L") {
        if( document.title === "Compare result sets" ) {
            let select_element = document.getElementById('id_left_set');
            summary_string = select_element.options[select_element.selectedIndex].value;
        } else if( document.title === "Result set comparisons" ) {
            summary_string += document.getElementById('id_left_comp').value.toLowerCase();
            summary_string += document.getElementById('id_left_parcel').value.toLowerCase()[0];
            summary_string += document.getElementById('id_left_split').value.toLowerCase()[0];
            summary_string += document.getElementById('id_left_train_mask').value;
            summary_string += document.getElementById('id_left_algo').value;
        }
    } else if (side === "R") {
        if( document.title === "Compare result sets" ) {
            let select_element = document.getElementById('id_right_set');
            summary_string = select_element.options[select_element.selectedIndex].value;
        } else if( document.title === "Result set comparisons" ) {
            summary_string += document.getElementById('id_right_comp').value.toLowerCase();
            summary_string += document.getElementById('id_right_parcel').value.toLowerCase()[0];
            summary_string += document.getElementById('id_right_split').value.toLowerCase()[0];
            summary_string += document.getElementById('id_right_train_mask').value;
            summary_string += document.getElementById('id_right_algo').value;
        }
    } else if (side === "center_set_string") {
        if( document.title === "Result set performance" ) {
            summary_string += document.getElementById('id_comp').value.toLowerCase();
            summary_string += document.getElementById('id_parcel').value.toLowerCase()[0];
            summary_string += document.getElementById('id_split').value.toLowerCase()[0];
            summary_string += document.getElementById('id_train_mask').value;
            summary_string += document.getElementById('id_algo').value;
        }
    }
    console.log("    calculated " + side + " summary_string of '" + summary_string + "'");
    return summary_string;
}

function latestUpdateState(elementId) {
    let request = new XMLHttpRequest();
    request.onreadystatechange = function () {
        if (request.readyState === 4 && request.status === 200) {
            let responseJsonObj = JSON.parse(this.responseText);
            let response = "";
            response += responseJsonObj.num_results + " total results: ";
            response += responseJsonObj.num_actuals + " actuals, ";
            response += responseJsonObj.num_shuffles + "+";
            response += responseJsonObj.num_distshuffles + "+";
            response += responseJsonObj.num_edgeshuffles + " permutations";
            response += "<br />";
            response += "last refreshed " + responseJsonObj.summary_date;
            document.getElementById(elementId).innerHTML = response;
        }
    };
    request.open("GET", "/gedata/REST/latest/", true);
    request.send();
}

function initUi() {
    console.log("  initializing GE Data Manager UI (in main.js)");

    // Fill in the footer's status span
    latestUpdateState("latest_result_summary");

    // For the compare.html and comparison.html views:
    //buildPlot('left_image', 'left_set_string');
    //buildPlot('right_image', 'right_set_string');
    if( document.title === "Result set comparisons" ) {
        console.log( "    adding event listeners to form elements.")
        //document.getElementById("id_left_comp").addEventListener('change', buildPlot('left_image', 'left_set_string'));
        //document.getElementById("id_left_parcel").addEventListener('change', buildPlot('left_image', 'left_set_string'));
        //document.getElementById("id_left_split").addEventListener('change', buildPlot('left_image', 'left_set_string'));
        //document.getElementById("id_left_train_mask").addEventListener('change', buildPlot('left_image', 'left_set_string'));
        //document.getElementById("id_left_algo").addEventListener('change', buildPlot('left_image', 'left_set_string'));
        //document.getElementById("id_right_comp").addEventListener('change', buildPlot('right_image', 'right_set_string'));
        //document.getElementById("id_right_parcel").addEventListener('change', buildPlot('right_image', 'right_set_string'));
        //document.getElementById("id_right_split").addEventListener('change', buildPlot('right_image', 'right_set_string'));
        //document.getElementById("id_right_train_mask").addEventListener('change', buildPlot('right_image', 'right_set_string'));
        //document.getElementById("id_right_algo").addEventListener('change', buildPlot('right_image', 'right_set_string'));
    } else if ( document.title === "Compare result sets" ) {
        document.getElementById("id_left_set").addEventListener('change', buildPlot('left_image', 'id_left_set'));
        document.getElementById("id_right_set").addEventListener('change', buildPlot('right_image', 'id_right_set'));
    }

    // This is supposed to manage highlighting the active menu item, but doesn't work. One day I'll debug it.

    $(".nav-item .nav-link").on("click", function(){
        $(".nav-item").find(".active").removeClass("active");
        $(this).addClass("active");
    });

    console.log("  completed initing GE Data Manager UI (in main.js)");
}

function inventory_td_contents(jsonObject) {

    // First, set up the table cell with a few extra spans. We can fill them in later.
    let cell_text = jsonObject.none + " (" + jsonObject.agno + "+" + jsonObject.dist + "+" + jsonObject.edge + ")";
    document.getElementById(jsonObject.signature).innerHTML = "<p>" + cell_text +
        " <span id=\"" + jsonObject.signature + "tt\"></span>" +
        " <span id=\"" + jsonObject.signature + "dna\"></span>" +
        " <span id=\"" + jsonObject.signature + "pf\"></span>" +
        "</p>";

    // Fill in the train_test_ span.
    let ttString = "";
    let ttUrl = "/static/gedata/plots/train_test_" + jsonObject.signature + ".png";
    let ttRequest = new XMLHttpRequest();
    ttRequest.onreadystatechange = function() {
        if(ttRequest.readyState === 4) {
            if(ttRequest.status === 200) {
                ttString = "<a href=\"" + ttUrl + "\" target=\"blank\"><i class=\"fas fa-box-up\"></i></a>";
            }
            if(ttRequest.status === 404) {
                ttString = "<i class=\"fal fa-box-up\"></i>";
            }
            document.getElementById(jsonObject.signature + "tt").innerHTML = ttString;
        }
    };
    ttRequest.open('HEAD', ttUrl, true);
    ttRequest.send();

    // Fill in the performance_ span.
    let pfString = "";
    let pfUrl = "/static/gedata/plots/performance_" + jsonObject.signature + ".png";
    let pfRequest = new XMLHttpRequest();
    pfRequest.onreadystatechange = function() {
        if(pfRequest.readyState === 4) {
            if(pfRequest.status === 200) {
                pfString = "<a href=\"" + pfUrl + "\" target=\"blank\"><i class=\"fas fa-chart-line\"></i></a>";
            }
            if(pfRequest.status === 404) {
                pfString = "<i class=\"fal fa-chart-line-down\"></i>";
            }
            document.getElementById(jsonObject.signature + "pf").innerHTML = pfString;
        }
    };
    pfRequest.open('HEAD', pfUrl, true);
    pfRequest.send();

    // Fill in the gene_list_ span.
    let dnaString = "";
    let dnaUrl = "/static/gedata/plots/train_test_" + jsonObject.signature + ".html";
    let dnaRequest = new XMLHttpRequest();
    dnaRequest.onreadystatechange = function() {
        if(dnaRequest.readyState === 4) {
            if(dnaRequest.status === 200) {
                dnaString = "<a href=\"" + dnaUrl + "\" target=\"blank\"><i class=\"fas fa-dna\"></i></a>";
            }
            if(dnaRequest.status === 404) {
                dnaString = "<i class=\"fal fa-dna\"></i>";
            }
            document.getElementById(jsonObject.signature + "dna").innerHTML = dnaString;
        }
    };
    dnaRequest.open('HEAD', dnaUrl, true);
    dnaRequest.send();

}

function fillInventoryTable() {
    let pby, sby, comp, mask;
    let ps = ["w", "g"];
    let ss = ["w", "g"];
    let comps = ["hcp", "nki"];
    let masks = ["00", "16", "32", "64"];

    for(pby = 0; pby < ps.length; ++pby) {
        for(sby = 0; sby < ss.length; ++sby) {
            for(comp = 0; comp < comps.length; ++comp) {
                for(mask = 0; mask < masks.length; ++mask) {
                    let idString = comps[comp] + ps[pby] + ss[sby] + masks[mask] + "s";
                    // The ajax request asks django to query the database for it.
                    let request = new XMLHttpRequest();
                    request.onreadystatechange = function () {
                        if (request.readyState === 4 && request.status === 200) {
                            let responseJsonObj = JSON.parse(this.responseText);
                            if (responseJsonObj.signature === idString) {
                                inventory_td_contents(responseJsonObj);
                            }
                        }
                    };
                    request.open("GET", "/gedata/REST/inventory/" + idString, true);
                    request.send();
                }
            }
        }
    }

}