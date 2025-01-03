import { getE, getV, clearE, showE, addEvent } from "/js/dom_utils.js";
import { runApp, getAppProgress, getAppResult, removeApp } from "/js/service.js";

let attackStage = 0;

async function checkAttackProgress() {
    const progressInfo = await getAppProgress("mu-mia");
    //console.log(progressInfo);
    let msg = "";
    if (progressInfo.status == "running") {
        if (attackStage == 0) {
            msg = "학습 모델에 대한 멤버쉽 추론 공격을 실행 중입니다.<br>";
        }
        else {
            msg = "언러닝 모델에 대한 멤버쉽 추론 공격을 실행 중입니다.<br>";
        }
    }
    if (progressInfo.status == "done") {
        msg = "멤버쉽 추론 공격을 완료했습니다.<br>";
    }
    if (progressInfo.status == "none") {
        msg = "멤버쉽 추론 공격이 완료되었습니다.<br>";
    }

    if (progressInfo.message) {
        msg += `${progressInfo.message.replaceAll("\n", "<br>")}<br>`;
    }

    getE("attack_output_log").innerHTML = msg;
    if (progressInfo.status != "running") {
        const result = await getAppResult("mu-mia");
        console.log(result);
        await removeApp("mu-mia");

        if (attackStage == 0) {
            getE("before_acc").innerText = `${(result.acc * 100).toFixed(2)}%`;
            setTimeout(() => ADP.show(getE("card_before_acc"), "flip-right"), 500);

            const attackType = getV("select_attack");
            await runApp("mu-mia", {
                op_mode: "mia-unlearn",
                data_type: "kr_celeb",
                forget_class_idx: 9,
                n_classes: 10,
                attack_type: attackType
            });
            attackStage = 1;
            setTimeout(checkAttackProgress, 1000);
        }
        else if (attackStage == 1) {
            getE("after_acc").innerText = `${(result.retain_acc * 100).toFixed(2)}%`;
            setTimeout(() => ADP.show(getE("card_after_acc"), "flip-right"), 500);

            getE("before_forget_acc").innerText = `100.00%`;
            setTimeout(() => ADP.show(getE("card_before_forget_acc"), "flip-right"), 500);

            getE("after_forget_acc").innerText = `${(result.forget_acc * 100).toFixed(2)}%`;
            setTimeout(() => ADP.show(getE("card_after_forget_acc"), "flip-right"), 700);
        }

        //clearE("attack_output_log")
        //showE("attack_graph");
        //drawPopulationRocGraph(result);
    }
    else {
        setTimeout(checkAttackProgress, 1000);
    }
}

async function execAttack() {
    const attackType = getV("select_attack");

    attackStage = 0;
    await runApp("mu-mia", {
        op_mode: "attack-test",
        data_type: "kr_celeb",
        forget_class_idx: 9,
        n_classes: 10,
        attack_type: attackType
    });
    setTimeout(checkAttackProgress, 1000);

    clearE("attack_output_log")
    showE("attack_output")

    ADP.hide(getE("card_before_acc"), "flip-right");
    ADP.hide(getE("card_after_acc"), "flip-right");
    ADP.hide(getE("card_before_forget_acc"), "flip-right");
    ADP.hide(getE("card_after_forget_acc"), "flip-right");
}

async function init() {
    addEvent("btn_exec_attack", "click", execAttack);
}

init();